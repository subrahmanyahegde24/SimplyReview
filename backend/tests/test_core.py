from simplyreview import sparse
from simplyreview.documents import CHUNK_CHARS, chunk_pages
from simplyreview.extraction import ChunkGraph, Entity, Relation, entity_key
from simplyreview.graph_store import flatten_graph
from simplyreview.rag import build_context
from simplyreview.vector_store import ChunkHit


def test_tokenize_drops_stopwords_and_folds_plurals():
    assert sparse.tokenize("The Transformers use attention layers on ImageNet-1k") == [
        "transformer", "attention", "layer", "imagenet-1k",
    ]


def test_token_index_is_stable_and_31_bit():
    assert sparse.token_index("bert") == sparse.token_index("bert")
    assert 0 <= sparse.token_index("bert") < 2**31


def test_bm25_document_weights_saturate_with_term_frequency():
    idx, once = sparse.encode_document("graph retrieval")
    _, thrice = sparse.encode_document("graph graph graph retrieval")
    graph = sparse.token_index("graph")
    w_once = once[idx.index(graph)]
    w_thrice = thrice[sparse.encode_document("graph graph graph retrieval")[0].index(graph)]
    assert w_once < w_thrice < 3 * w_once
    assert w_thrice < sparse.K1 + 1


def test_query_encoding_is_binary_and_sorted():
    indices, values = sparse.encode_query("graph RAG graph")
    assert indices == sorted(indices)
    assert values == [1.0] * len(indices)
    assert sparse.encode_query("the of and") == ([], [])


def test_chunking_keeps_pages_and_strips_references():
    body = "Graph retrieval improves multi-hop QA. " * 80
    pages = ["Title page " + body, body, "Conclusion. " + body + "\nReferences\n[1] Someone et al. 2020."]
    chunks = chunk_pages("p1", pages)
    assert {c.page for c in chunks} == {1, 2, 3}
    assert all(len(c.text) <= CHUNK_CHARS + 200 for c in chunks)
    assert not any("Someone et al" in c.text for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "p1:0"


def test_entity_key_merges_surface_variants():
    assert entity_key("BERT-Base") == entity_key("bert base") == "bert base"


def test_flatten_graph_dedupes_and_backfills_missing_endpoints():
    from simplyreview.documents import Chunk

    chunk = Chunk(id="p1:0", paper_id="p1", index=0, page=1, text="...")
    graph = ChunkGraph(
        chunk=0,
        entities=[Entity(name="BERT", type="Model", description="A transformer encoder")],
        relations=[
            Relation(source="BERT", target="SQuAD", type="EVALUATED_ON", description="F1 93.2 on SQuAD"),
            Relation(source="BERT", target="bert", type="RELATED_TO", description="self loop"),
        ],
    )
    mentions, relations = flatten_graph("p1", [(chunk, graph), (chunk, graph)])
    assert sorted(m["key"] for m in mentions) == ["bert", "squad"]
    assert next(m for m in mentions if m["key"] == "squad")["type"] == "Concept"
    assert len(relations) == 1 and relations[0]["type"] == "EVALUATED_ON"


def _hit(chunk_id: str) -> ChunkHit:
    return ChunkHit(chunk_id=chunk_id, paper_id="p1", paper_title="Paper", page=2, text=f"text {chunk_id}",
                    score=0.5, dense_rank=1, keyword_rank=None)


def test_build_context_numbers_sources_once_and_cites_graph_facts():
    graph = {
        "seeds": [],
        "evidence": [
            {"chunk_id": "p1:1", "paper_id": "p1", "paper_title": "Paper", "page": 2, "text": "dup"},
            {"chunk_id": "p2:0", "paper_id": "p2", "paper_title": "Other", "page": 1, "text": "graph only"},
        ],
        "relations": [
            {"source_name": "BERT", "target_name": "SQuAD", "type": "EVALUATED_ON", "description": "F1 93.2",
             "chunks": ["p2:0", "p9:9"], "papers": 2},
        ],
    }
    sources, context = build_context([_hit("p1:0"), _hit("p1:1")], graph)
    assert [(s["n"], s["origin"]) for s in sources] == [(1, "vector"), (2, "both"), (3, "graph")]
    assert "BERT --EVALUATED_ON--> SQuAD: F1 93.2 (in 2 papers) [3]" in context


def test_build_context_without_graph():
    sources, context = build_context([_hit("p1:0")], None)
    assert len(sources) == 1 and "KNOWLEDGE GRAPH" not in context


def test_admin_gate_fails_closed_on_vercel(monkeypatch):
    import pytest
    from fastapi import HTTPException

    from simplyreview import api
    from simplyreview.config import get_settings

    base = get_settings()
    cases = [
        (dict(admin_token=None, on_vercel=False), None, None),  # local dev stays open
        (dict(admin_token=None, on_vercel=True), None, 503),
        (dict(admin_token="s3cret", on_vercel=True), "wrong", 401),
        (dict(admin_token="s3cret", on_vercel=True), "s3cret", None),
    ]
    for overrides, token, status in cases:
        monkeypatch.setattr(api, "get_settings", lambda o=overrides: base.__class__(**{**base.__dict__, **o}))
        if status is None:
            api._require_admin(token)
        else:
            with pytest.raises(HTTPException) as err:
                api._require_admin(token)
            assert err.value.status_code == status
