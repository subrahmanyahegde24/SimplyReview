"""Query pipeline: Qdrant hybrid search and Neo4j graph search in parallel, fused into one cited answer."""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import asdict

from . import graph_store, llm, vector_store
from .ingest import ensure_ready
from .vector_store import ChunkHit

ENTITY_MIN_SCORE = 0.62
MAX_SEEDS = 8
GRAPH_EVIDENCE = 4
SNIPPET_CHARS = 600

SYSTEM = """You are SimplyReview, a research assistant answering questions about the user's library of papers.

You receive two kinds of evidence:
1. SOURCES: numbered passages retrieved from the papers.
2. KNOWLEDGE GRAPH: entity relations extracted across all papers (which method uses which dataset,
   what outperforms what, and so on). These reveal connections no single passage states.

Rules:
- Answer only from this evidence. If it is not enough, say exactly what is missing.
- Cite passages inline as [n] or [n, m]. A graph fact is cited by the passage numbers attached to it;
  never write [KG] or cite the graph itself.
- When the graph links ideas across papers, say so explicitly; that synthesis is the point.
- Be concise. Use markdown: short paragraphs, bullet lists, and a table when comparing things."""


async def graph_retrieve(question: str, dense: list[float]) -> dict:
    semantic, keyword = await asyncio.gather(
        vector_store.search_entities(dense, k=MAX_SEEDS, min_score=ENTITY_MIN_SCORE),
        graph_store.fulltext_entities(question, k=5),
    )
    seeds: dict[str, dict] = {}
    for hit in semantic:
        seeds[hit.key] = {"key": hit.key, "name": hit.name, "type": hit.type, "score": round(hit.score, 3), "via": "semantic"}
    for row in keyword:
        if row["key"] in seeds:
            seeds[row["key"]]["via"] = "both"
        else:
            seeds[row["key"]] = {**row, "score": round(row["score"], 3), "via": "keyword"}
    seed_list = sorted(seeds.values(), key=lambda s: s["via"] == "both", reverse=True)[:MAX_SEEDS]
    keys = [s["key"] for s in seed_list]

    relations, evidence = await asyncio.gather(graph_store.expand(keys), graph_store.evidence_chunks(keys, GRAPH_EVIDENCE))
    return {"seeds": seed_list, "relations": relations, "evidence": evidence}


def subgraph(graph: dict) -> dict:
    nodes = {s["key"]: {"id": s["key"], "name": s["name"], "type": s["type"], "seed": True} for s in graph["seeds"]}
    for rel in graph["relations"]:
        for side in ("source", "target"):
            nodes.setdefault(
                rel[side], {"id": rel[side], "name": rel[f"{side}_name"], "type": rel[f"{side}_type"], "seed": False}
            )
    links = [
        {"source": r["source"], "target": r["target"], "type": r["type"], "description": r["description"], "bridge": r["bridge"]}
        for r in graph["relations"]
    ]
    return {"nodes": list(nodes.values()), "links": links}


def build_context(hits: list[ChunkHit], graph: dict | None) -> tuple[list[dict], str]:
    """Number every passage once, whichever retriever found it, then render the prompt context."""
    sources: list[dict] = []
    by_chunk: dict[str, dict] = {}

    def add(chunk_id, paper_id, title, page, text, origin):
        if chunk_id in by_chunk:
            if by_chunk[chunk_id]["origin"] != origin:
                by_chunk[chunk_id]["origin"] = "both"
            return
        source = {"n": len(sources) + 1, "chunk_id": chunk_id, "paper_id": paper_id, "paper_title": title,
                  "page": page, "text": text, "origin": origin}
        sources.append(source)
        by_chunk[chunk_id] = source

    for hit in hits:
        add(hit.chunk_id, hit.paper_id, hit.paper_title, hit.page, hit.text, "vector")
    if graph:
        for ev in graph["evidence"]:
            add(ev["chunk_id"], ev["paper_id"], ev["paper_title"], ev["page"], ev["text"], "graph")

    parts = ["SOURCES"]
    for s in sources:
        parts.append(f"[{s['n']}] {s['paper_title']} (p.{s['page']})\n{s['text']}")

    if graph and graph["relations"]:
        parts.append("KNOWLEDGE GRAPH")
        for rel in graph["relations"]:
            cites = "".join(f"[{by_chunk[c]['n']}]" for c in rel["chunks"] if c in by_chunk)
            spread = f" (in {rel['papers']} papers)" if rel["papers"] > 1 else ""
            parts.append(
                f"- {rel['source_name']} --{rel['type']}--> {rel['target_name']}: {rel['description']}{spread} {cites}".rstrip()
            )
    return sources, "\n\n".join(parts)


def _ms(since: float) -> int:
    return round((time.perf_counter() - since) * 1000)


async def answer(question: str, top_k: int = 6) -> AsyncIterator[dict]:
    started = time.perf_counter()
    timings: dict[str, int] = {}
    await ensure_ready()

    yield {"type": "stage", "stage": "embed", "status": "running"}
    t = time.perf_counter()
    dense = await llm.embed_query(question)
    timings["embed"] = _ms(t)
    yield {"type": "stage", "stage": "embed", "status": "done", "ms": timings["embed"]}

    async def timed(name, coro):
        t0 = time.perf_counter()
        try:
            return name, await coro, None
        except Exception as exc:
            return name, None, str(exc)
        finally:
            timings[name] = _ms(t0)

    yield {"type": "stage", "stage": "vector", "status": "running"}
    yield {"type": "stage", "stage": "graph", "status": "running"}
    tasks = [
        asyncio.create_task(timed("vector", vector_store.hybrid_search(question, dense, top_k))),
        asyncio.create_task(timed("graph", graph_retrieve(question, dense))),
    ]
    hits: list[ChunkHit] = []
    graph: dict | None = None
    async for task in asyncio.as_completed(tasks):
        name, result, error = await task
        if error:
            yield {"type": "stage", "stage": name, "status": "error", "ms": timings[name], "error": error}
            continue
        if name == "vector":
            hits = result
            yield {"type": "vector", "hits": [{**asdict(h), "text": h.text[:SNIPPET_CHARS]} for h in hits]}
        else:
            graph = result
            yield {"type": "graph", "seeds": graph["seeds"], "relations": graph["relations"],
                   "evidence": [{**e, "text": e["text"][:SNIPPET_CHARS]} for e in graph["evidence"]],
                   "subgraph": subgraph(graph)}
        yield {"type": "stage", "stage": name, "status": "done", "ms": timings[name]}

    t = time.perf_counter()
    sources, context = build_context(hits, graph)
    yield {"type": "sources", "sources": [{**s, "text": s["text"][:SNIPPET_CHARS]} for s in sources]}
    timings["fuse"] = _ms(t)

    if not sources:
        yield {"type": "token", "text": "I couldn't find anything relevant in your library. Try uploading papers on this topic first."}
        yield {"type": "done", "timings": {**timings, "total": _ms(started)}}
        return

    yield {"type": "stage", "stage": "generate", "status": "running"}
    t = time.perf_counter()
    first_token: int | None = None
    async for text in llm.stream_answer(SYSTEM, f"{context}\n\nQUESTION: {question}"):
        if first_token is None:
            first_token = _ms(t)
        yield {"type": "token", "text": text}
    timings["generate"] = _ms(t)
    yield {"type": "stage", "stage": "generate", "status": "done", "ms": timings["generate"], "first_token_ms": first_token}
    yield {"type": "done", "timings": {**timings, "total": _ms(started)}}
