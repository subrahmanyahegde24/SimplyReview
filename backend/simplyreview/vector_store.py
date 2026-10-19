"""Qdrant: hybrid (dense + BM25 sparse) chunk search and semantic entity linking."""

import uuid
from dataclasses import dataclass
from functools import cache

from . import sparse
from .config import get_settings
from .documents import Chunk

DENSE = "dense"
BM25 = "bm25"
PREFETCH = 30
UPSERT_BATCH = 64


@dataclass
class ChunkHit:
    chunk_id: str
    paper_id: str
    paper_title: str
    page: int
    text: str
    score: float
    dense_rank: int | None
    keyword_rank: int | None


@dataclass
class EntityHit:
    key: str
    name: str
    type: str
    score: float


@cache
def _client():
    from qdrant_client import AsyncQdrantClient

    settings = get_settings()
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


async def ensure_collections() -> None:
    from qdrant_client import models

    settings = get_settings()
    client = _client()
    dense = models.VectorParams(size=settings.embed_dim, distance=models.Distance.COSINE)

    if not await client.collection_exists(settings.chunks_collection):
        await client.create_collection(
            settings.chunks_collection,
            vectors_config={DENSE: dense},
            sparse_vectors_config={BM25: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        await client.create_payload_index(
            settings.chunks_collection, field_name="paper_id", field_schema=models.PayloadSchemaType.KEYWORD
        )
    if not await client.collection_exists(settings.entities_collection):
        await client.create_collection(settings.entities_collection, vectors_config={DENSE: dense})


def _paper_filter(paper_id: str):
    from qdrant_client import models

    return models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))])


async def paper_exists(paper_id: str) -> bool:
    result = await _client().count(get_settings().chunks_collection, count_filter=_paper_filter(paper_id), exact=False)
    return result.count > 0


async def upsert_chunks(chunks: list[Chunk], title: str, vectors: list[list[float]]) -> None:
    from qdrant_client import models

    points = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        indices, values = sparse.encode_document(chunk.text)
        points.append(
            models.PointStruct(
                id=_point_id(chunk.id),
                vector={DENSE: vector, BM25: models.SparseVector(indices=indices, values=values)},
                payload={
                    "chunk_id": chunk.id,
                    "paper_id": chunk.paper_id,
                    "paper_title": title,
                    "page": chunk.page,
                    "index": chunk.index,
                    "text": chunk.text,
                },
            )
        )
    for i in range(0, len(points), UPSERT_BATCH):
        await _client().upsert(get_settings().chunks_collection, points=points[i : i + UPSERT_BATCH])


async def upsert_entities(entities: list[dict], vectors: list[list[float]]) -> None:
    from qdrant_client import models

    points = [
        models.PointStruct(id=_point_id(e["key"]), vector={DENSE: v}, payload=e)
        for e, v in zip(entities, vectors, strict=True)
    ]
    for i in range(0, len(points), UPSERT_BATCH):
        await _client().upsert(get_settings().entities_collection, points=points[i : i + UPSERT_BATCH])


async def hybrid_search(query: str, dense: list[float], k: int) -> list[ChunkHit]:
    """One round trip: dense-only, BM25-only, and their RRF fusion.

    The fused list is the result; the single-mode lists only annotate each hit with
    its rank under each retriever so the dashboard can show why it was retrieved.
    """
    from qdrant_client import models

    settings = get_settings()
    indices, values = sparse.encode_query(query)
    has_keywords = bool(indices)
    sparse_query = models.SparseVector(indices=indices, values=values)

    prefetch = [models.Prefetch(query=dense, using=DENSE, limit=PREFETCH)]
    requests = [models.QueryRequest(query=dense, using=DENSE, limit=PREFETCH)]
    if has_keywords:
        prefetch.append(models.Prefetch(query=sparse_query, using=BM25, limit=PREFETCH))
        requests.append(models.QueryRequest(query=sparse_query, using=BM25, limit=PREFETCH))
    requests.append(
        models.QueryRequest(
            prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k, with_payload=True
        )
    )

    responses = await _client().query_batch_points(settings.chunks_collection, requests=requests)
    dense_ranks = {p.id: rank for rank, p in enumerate(responses[0].points, start=1)}
    keyword_ranks = {p.id: rank for rank, p in enumerate(responses[1].points, start=1)} if has_keywords else {}

    return [
        ChunkHit(
            chunk_id=p.payload["chunk_id"],
            paper_id=p.payload["paper_id"],
            paper_title=p.payload["paper_title"],
            page=p.payload["page"],
            text=p.payload["text"],
            score=p.score,
            dense_rank=dense_ranks.get(p.id),
            keyword_rank=keyword_ranks.get(p.id),
        )
        for p in responses[-1].points
    ]


async def search_entities(dense: list[float], k: int, min_score: float) -> list[EntityHit]:
    response = await _client().query_points(
        get_settings().entities_collection, query=dense, using=DENSE, limit=k, score_threshold=min_score
    )
    return [EntityHit(key=p.payload["key"], name=p.payload["name"], type=p.payload["type"], score=p.score) for p in response.points]


async def delete_paper(paper_id: str) -> None:
    from qdrant_client import models

    await _client().delete(get_settings().chunks_collection, points_selector=models.FilterSelector(filter=_paper_filter(paper_id)))


async def delete_entities(keys: list[str]) -> None:
    if keys:
        await _client().delete(get_settings().entities_collection, points_selector=[_point_id(k) for k in keys])
