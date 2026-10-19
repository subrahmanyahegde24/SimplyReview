"""PDF ingestion pipeline. Yields progress events so the dashboard can show each stage live."""

import asyncio
import time
from collections.abc import AsyncIterator

from . import graph_store, llm, vector_store
from .documents import chunk_pages, paper_id_for, parse_pdf
from .extraction import CHUNKS_PER_CALL, extract_graph, extract_meta

EXTRACT_CONCURRENCY = 6

_ready = False
_ready_lock = asyncio.Lock()


async def ensure_ready() -> None:
    """Create collections, constraints and indexes once per process."""
    global _ready
    if _ready:
        return
    async with _ready_lock:
        if not _ready:
            await asyncio.gather(vector_store.ensure_collections(), graph_store.ensure_schema())
            _ready = True


async def delete_paper(paper_id: str) -> None:
    orphans = await graph_store.delete_paper(paper_id)
    await asyncio.gather(vector_store.delete_paper(paper_id), vector_store.delete_entities(orphans))


async def ingest_pdf(data: bytes, filename: str) -> AsyncIterator[dict]:
    started = time.perf_counter()
    paper_id = paper_id_for(data)
    await ensure_ready()

    if await vector_store.paper_exists(paper_id):
        yield {"type": "skipped", "paper_id": paper_id, "reason": "This paper is already in the library."}
        return

    try:
        yield {"type": "stage", "stage": "parse", "status": "running"}
        parsed = await asyncio.to_thread(parse_pdf, data)
        chunks = chunk_pages(paper_id, parsed.pages)
        if not chunks:
            yield {"type": "error", "message": "No extractable text found. Is this a scanned PDF?"}
            return
        yield {"type": "stage", "stage": "parse", "status": "done", "pages": len(parsed.pages), "chunks": len(chunks)}

        yield {"type": "stage", "stage": "embed", "status": "running"}
        meta, vectors = await asyncio.gather(
            extract_meta(parsed.pages), llm.embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
        )
        yield {"type": "meta", "paper_id": paper_id, **meta.model_dump()}
        await vector_store.upsert_chunks(chunks, meta.title, vectors)
        await graph_store.upsert_paper(paper_id, meta, filename, len(parsed.pages), chunks)
        yield {"type": "stage", "stage": "embed", "status": "done", "vectors": len(vectors)}

        batches = [chunks[i : i + CHUNKS_PER_CALL] for i in range(0, len(chunks), CHUNKS_PER_CALL)]
        yield {"type": "stage", "stage": "extract", "status": "running", "total": len(batches)}
        semaphore = asyncio.Semaphore(EXTRACT_CONCURRENCY)

        async def run(batch):
            async with semaphore:
                return await extract_graph(meta.title, batch)

        extracted = []
        failed = 0
        tasks = [asyncio.create_task(run(b)) for b in batches]
        async for task in asyncio.as_completed(tasks):
            try:
                extracted.extend(await task)
            except Exception:
                failed += 1
            done = len([t for t in tasks if t.done()])
            entities = len({e.name for _, g in extracted for e in g.entities})
            yield {"type": "progress", "stage": "extract", "done": done, "total": len(batches), "entities": entities}

        mentions, relations = graph_store.flatten_graph(paper_id, extracted)
        await graph_store.upsert_graph(paper_id, mentions, relations)
        yield {"type": "stage", "stage": "extract", "status": "done", "failed_batches": failed}

        yield {"type": "stage", "stage": "link", "status": "running"}
        unique = {m["key"]: m for m in mentions}
        entities = [{"key": k, "name": m["name"], "type": m["type"], "description": m["description"]} for k, m in unique.items()]
        if entities:
            entity_vectors = await llm.embed(
                [f"{e['name']} ({e['type']}): {e['description']}" for e in entities], "RETRIEVAL_DOCUMENT"
            )
            await vector_store.upsert_entities(entities, entity_vectors)
        yield {"type": "stage", "stage": "link", "status": "done"}

        yield {
            "type": "done",
            "paper_id": paper_id,
            "title": meta.title,
            "chunks": len(chunks),
            "entities": len(entities),
            "relations": len(relations),
            "seconds": round(time.perf_counter() - started, 1),
        }
    except Exception as exc:
        # Leave no half-indexed paper behind; a retry then starts clean.
        await delete_paper(paper_id)
        yield {"type": "error", "message": f"Ingestion failed: {exc}"}
