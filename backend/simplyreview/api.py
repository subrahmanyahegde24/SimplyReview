"""HTTP API. Long-running work streams Server-Sent Events so the dashboard updates in real time."""

import json
import secrets
from collections.abc import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import graph_store, ingest, rag
from .config import get_settings

MAX_UPLOAD_BYTES = 4_400_000  # Vercel caps request bodies at 4.5 MB

app = FastAPI(title="SimplyReview API", docs_url="/api/docs", openapi_url="/api/openapi.json")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=15)


def _require_admin(token: str | None) -> None:
    """Fail closed when deployed: without ADMIN_TOKEN, only local development may write."""
    settings = get_settings()
    if not settings.admin_token:
        if settings.on_vercel:
            raise HTTPException(status_code=503, detail="Uploads are disabled until ADMIN_TOKEN is set on the deployment")
        return
    if not (token and secrets.compare_digest(token, settings.admin_token)):
        raise HTTPException(status_code=401, detail="Upload key required")


def _sse(events: AsyncIterator[dict]) -> StreamingResponse:
    async def body():
        try:
            async for event in events:
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            # Compression buffers the stream into bursts; tell every proxy to pass it through as-is.
            "Cache-Control": "no-cache, no-transform",
            "Content-Encoding": "identity",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/api/stats")
async def stats() -> dict:
    await ingest.ensure_ready()
    return await graph_store.stats()


@app.get("/api/papers")
async def papers() -> list[dict]:
    await ingest.ensure_ready()
    return await graph_store.list_papers()


@app.get("/api/graph")
async def graph(limit: int = 120) -> dict:
    await ingest.ensure_ready()
    return await graph_store.overview(min(max(limit, 10), 400))


@app.get("/api/config")
async def config() -> dict:
    settings = get_settings()
    return {"chat_model": settings.chat_model, "embed_model": settings.embed_model, "uploads_locked": bool(settings.admin_token) or settings.on_vercel}


@app.post("/api/query")
async def query(request: QueryRequest) -> StreamingResponse:
    return _sse(rag.answer(request.question.strip(), request.top_k))


@app.post("/api/ingest")
async def upload(file: UploadFile, x_admin_token: str | None = Header(default=None)) -> StreamingResponse:
    _require_admin(x_admin_token)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF too large (max 4.4 MB on the hosted app; use the CLI for bigger files)")
    return _sse(ingest.ingest_pdf(data, file.filename))


@app.delete("/api/papers/{paper_id}")
async def delete(paper_id: str, x_admin_token: str | None = Header(default=None)) -> dict:
    _require_admin(x_admin_token)
    await ingest.delete_paper(paper_id)
    return {"deleted": paper_id}
