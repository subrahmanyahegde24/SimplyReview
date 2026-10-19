"""Neo4j knowledge graph.

Schema:
    (:Paper {id, title, authors, year, summary, filename, pages})
    (:Chunk {id, index, page, text})-[:PART_OF]->(:Paper)
    (:Chunk)-[:MENTIONS]->(:Entity {key, name, type, description, mentions})
    (:Paper)-[:DISCUSSES {count}]->(:Entity)
    (:Entity)-[:RELATES {type, descriptions, chunks, papers, weight}]->(:Entity)
"""

import re
from functools import cache
from typing import Any

from .config import get_settings
from .documents import Chunk
from .extraction import ChunkGraph, PaperMeta, entity_key

SCHEMA = [
    "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
]

_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


@cache
def _driver():
    from neo4j import AsyncGraphDatabase

    settings = get_settings()
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
        notifications_min_severity="OFF",  # e.g. "relationship type does not exist" on an empty graph
    )


async def _run(query: str, read: bool = False, **params: Any) -> list[dict]:
    from neo4j import RoutingControl

    result = await _driver().execute_query(
        query,
        params,
        database_=get_settings().neo4j_database,
        routing_=RoutingControl.READ if read else RoutingControl.WRITE,
    )
    return [record.data() for record in result.records]


async def ensure_schema() -> None:
    for statement in SCHEMA:
        await _run(statement)


async def upsert_paper(paper_id: str, meta: PaperMeta, filename: str, pages: int, chunks: list[Chunk]) -> None:
    await _run(
        """
        MERGE (p:Paper {id: $id})
        SET p.title = $title, p.authors = $authors, p.year = $year, p.summary = $summary,
            p.filename = $filename, p.pages = $pages, p.created_at = coalesce(p.created_at, datetime())
        WITH p
        UNWIND $chunks AS chunk
        MERGE (c:Chunk {id: chunk.id})
        SET c.index = chunk.index, c.page = chunk.page, c.text = chunk.text
        MERGE (c)-[:PART_OF]->(p)
        """,
        id=paper_id,
        title=meta.title,
        authors=meta.authors,
        year=meta.year,
        summary=meta.summary,
        filename=filename,
        pages=pages,
        chunks=[{"id": c.id, "index": c.index, "page": c.page, "text": c.text} for c in chunks],
    )


def flatten_graph(paper_id: str, extracted: list[tuple[Chunk, ChunkGraph]]) -> tuple[list[dict], list[dict]]:
    """Turn per-chunk LLM output into deduplicated entity mentions and relation rows."""
    mentions: dict[tuple[str, str], dict] = {}
    relations: dict[tuple[str, str, str, str], dict] = {}

    known = {entity_key(e.name): e for _, graph in extracted for e in graph.entities}

    for chunk, graph in extracted:
        for entity in graph.entities:
            if key := entity_key(entity.name):
                mentions[(key, chunk.id)] = {
                    "key": key, "name": entity.name.strip(), "type": entity.type,
                    "description": entity.description, "chunk_id": chunk.id,
                }
        for rel in graph.relations:
            source, target = entity_key(rel.source), entity_key(rel.target)
            if not source or not target or source == target:
                continue
            for key, name in ((source, rel.source), (target, rel.target)):
                if (key, chunk.id) not in mentions:
                    entity = known.get(key)
                    mentions[(key, chunk.id)] = {
                        "key": key,
                        "name": entity.name.strip() if entity else name.strip(),
                        "type": entity.type if entity else "Concept",
                        "description": entity.description if entity else rel.description,
                        "chunk_id": chunk.id,
                    }
            relations[(source, target, rel.type, chunk.id)] = {
                "source": source, "target": target, "type": rel.type,
                "description": rel.description, "chunk_id": chunk.id, "paper_id": paper_id,
            }
    return list(mentions.values()), list(relations.values())


async def upsert_graph(paper_id: str, mentions: list[dict], relations: list[dict]) -> None:
    await _run(
        """
        UNWIND $mentions AS m
        MERGE (e:Entity {key: m.key})
        ON CREATE SET e.name = m.name, e.type = m.type, e.description = m.description, e.mentions = 0
        SET e.mentions = e.mentions + 1
        WITH e, m
        MATCH (c:Chunk {id: m.chunk_id})
        MERGE (c)-[:MENTIONS]->(e)
        """,
        mentions=mentions,
    )
    await _run(
        """
        UNWIND $relations AS rel
        MATCH (a:Entity {key: rel.source}), (b:Entity {key: rel.target})
        MERGE (a)-[r:RELATES {type: rel.type}]->(b)
        ON CREATE SET r.descriptions = [], r.chunks = [], r.papers = []
        SET r.chunks = CASE WHEN rel.chunk_id IN r.chunks THEN r.chunks ELSE r.chunks + rel.chunk_id END,
            r.papers = CASE WHEN rel.paper_id IN r.papers THEN r.papers ELSE r.papers + rel.paper_id END,
            r.descriptions = CASE WHEN size(r.descriptions) >= 5 OR rel.description IN r.descriptions
                                  THEN r.descriptions ELSE r.descriptions + rel.description END
        WITH r
        SET r.weight = size(r.chunks)
        """,
        relations=relations,
    )
    await _run(
        """
        MATCH (p:Paper {id: $id})<-[:PART_OF]-(:Chunk)-[:MENTIONS]->(e:Entity)
        WITH p, e, count(*) AS n
        MERGE (p)-[d:DISCUSSES]->(e)
        SET d.count = n
        """,
        id=paper_id,
    )


async def fulltext_entities(query: str, k: int) -> list[dict]:
    terms = [t for t in re.findall(r"[\w\-]+", query) if len(t) > 2]
    if not terms:
        return []
    lucene = " OR ".join(_LUCENE_SPECIAL.sub(r"\\\1", t) for t in terms)
    return await _run(
        """
        CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node, score
        RETURN node.key AS key, node.name AS name, node.type AS type, score
        ORDER BY score DESC LIMIT $k
        """,
        read=True,
        q=lucene,
        k=k,
    )


async def expand(seeds: list[str], per_seed: int = 6, limit: int = 30) -> list[dict]:
    """Relations around the seed entities, plus short paths that connect seeds to each other."""
    neighbourhood = await _run(
        """
        MATCH (s:Entity)-[r:RELATES]-(n:Entity) WHERE s.key IN $seeds
        WITH s, r, n ORDER BY r.weight DESC, n.mentions DESC
        WITH s, collect(r)[..$per_seed] AS rels
        UNWIND rels AS r
        RETURN DISTINCT elementId(r) AS id
        """,
        read=True,
        seeds=seeds,
        per_seed=per_seed,
    )
    bridges = await _run(
        """
        MATCH p = (a:Entity)-[:RELATES*1..2]-(b:Entity)
        WHERE a.key IN $seeds AND b.key IN $seeds AND elementId(a) < elementId(b)
        WITH p LIMIT 20
        UNWIND relationships(p) AS r
        RETURN DISTINCT elementId(r) AS id
        """,
        read=True,
        seeds=seeds,
    )
    bridge_ids = {row["id"] for row in bridges}
    ids = list(bridge_ids | {row["id"] for row in neighbourhood})
    if not ids:
        return []
    rows = await _run(
        """
        MATCH (a:Entity)-[r:RELATES]->(b:Entity) WHERE elementId(r) IN $ids
        RETURN a.key AS source, a.name AS source_name, a.type AS source_type,
               b.key AS target, b.name AS target_name, b.type AS target_type,
               r.type AS type, r.descriptions[0] AS description, r.chunks AS chunks,
               r.weight AS weight, size(r.papers) AS papers, elementId(r) AS id
        ORDER BY r.weight DESC LIMIT $limit
        """,
        read=True,
        ids=ids,
        limit=limit,
    )
    for row in rows:
        row["bridge"] = row.pop("id") in bridge_ids
    return rows


async def evidence_chunks(keys: list[str], k: int) -> list[dict]:
    """Chunks mentioning the most of the given entities: the graph's supporting text."""
    if not keys:
        return []
    return await _run(
        """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE e.key IN $keys
        WITH c, count(DISTINCT e) AS hits, collect(DISTINCT e.name) AS entities
        ORDER BY hits DESC, size(c.text) DESC LIMIT $k
        MATCH (c)-[:PART_OF]->(p:Paper)
        RETURN c.id AS chunk_id, c.text AS text, c.page AS page,
               p.id AS paper_id, p.title AS paper_title, hits, entities
        """,
        read=True,
        keys=keys,
        k=k,
    )


async def stats() -> dict:
    rows = await _run(
        """
        CALL () { MATCH (p:Paper) RETURN count(p) AS papers }
        CALL () { MATCH (c:Chunk) RETURN count(c) AS chunks }
        CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
        CALL () { MATCH ()-[r:RELATES]->() RETURN count(r) AS relations }
        RETURN papers, chunks, entities, relations
        """,
        read=True,
    )
    return rows[0]


async def list_papers() -> list[dict]:
    return await _run(
        """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:DISCUSSES]->(e:Entity)
        WITH p, count(e) AS entities
        RETURN p.id AS id, p.title AS title, p.authors AS authors, p.year AS year,
               p.summary AS summary, p.filename AS filename, p.pages AS pages, entities,
               toString(p.created_at) AS created_at
        ORDER BY p.created_at DESC
        """,
        read=True,
    )


async def overview(limit: int) -> dict:
    """The most-connected slice of the graph, for the dashboard visualisation."""
    nodes = await _run(
        """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)-[r:RELATES]-()
        WITH e, count(r) AS degree
        ORDER BY degree DESC, e.mentions DESC LIMIT $limit
        OPTIONAL MATCH (p:Paper)-[:DISCUSSES]->(e)
        RETURN e.key AS id, e.name AS name, e.type AS type, e.description AS description,
               e.mentions AS mentions, degree, collect(p.id) AS papers
        """,
        read=True,
        limit=limit,
    )
    keys = [n["id"] for n in nodes]
    links = await _run(
        """
        MATCH (a:Entity)-[r:RELATES]->(b:Entity) WHERE a.key IN $keys AND b.key IN $keys
        RETURN a.key AS source, b.key AS target, r.type AS type, r.descriptions[0] AS description, r.weight AS weight
        """,
        read=True,
        keys=keys,
    )
    return {"nodes": nodes, "links": links}


async def delete_paper(paper_id: str) -> list[str]:
    """Remove a paper and everything only it supported. Returns orphaned entity keys."""
    await _run(
        """
        MATCH ()-[r:RELATES]->() WHERE $id IN r.papers
        SET r.chunks = [c IN r.chunks WHERE NOT c STARTS WITH $prefix],
            r.papers = [p IN r.papers WHERE p <> $id]
        WITH r
        SET r.weight = size(r.chunks)
        WITH r WHERE r.weight = 0
        DELETE r
        """,
        id=paper_id,
        prefix=f"{paper_id}:",
    )
    await _run(
        """
        MATCH (p:Paper {id: $id})
        OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(p)
        DETACH DELETE c, p
        """,
        id=paper_id,
    )
    orphans = await _run(
        """
        MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-()
        WITH e, e.key AS key
        DETACH DELETE e
        RETURN key
        """
    )
    await _run(
        """
        MATCH (e:Entity)<-[m:MENTIONS]-()
        WITH e, count(m) AS n
        SET e.mentions = n
        """
    )
    return [row["key"] for row in orphans]
