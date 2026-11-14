"""Qdrant vector store: runs against a real server (docker compose up -d), skipped otherwise.
Uses a throwaway collection so it never touches the app's data."""
import uuid

import pytest

from app.config import ROOT, Settings
from app.search import dense
from app.search.index import KnowledgeBase, SearchConfig

from .conftest import TODAY


@pytest.fixture(scope="module")
def qdrant_settings(tmp_path_factory):
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333", timeout=3)
        client.get_collections()
    except Exception:  # noqa: BLE001
        pytest.skip("Qdrant not running (docker compose up -d)")
    if dense.get_model(Settings().dense_model) is None:
        pytest.skip("dense model not available")
    name = f"handbook_test_{uuid.uuid4().hex[:8]}"
    yield Settings(corpus_dir=ROOT / "data" / "corpus", uploads_dir=tmp_path_factory.mktemp("qup"),
                   use_dense=True, vector_store="qdrant", qdrant_collection=name)
    client.delete_collection(name)


def test_sync_embeds_once_then_reuses(qdrant_settings):
    first = KnowledgeBase(qdrant_settings).load()
    assert first.vector_store_name == "qdrant"
    n = len(first.snapshot.chunks)
    assert first.vector_store_stats == {"points": n, "embedded": n, "reused": 0, "deleted": 0}
    second = KnowledgeBase(qdrant_settings).load()
    assert second.vector_store_stats == {"points": n, "embedded": 0, "reused": n, "deleted": 0}


def test_qdrant_matches_in_memory_search(qdrant_settings, tmp_path):
    q = KnowledgeBase(qdrant_settings).load()
    m = KnowledgeBase(Settings(corpus_dir=ROOT / "data" / "corpus", uploads_dir=tmp_path,
                               use_dense=True, vector_store="memory")).load()
    for question in ["Am I allowed to install apps on my work machine?", "How long is the deploy freeze?"]:
        cfg = SearchConfig(use_bm25=False, lifecycle_rerank=False)
        assert [h.chunk.chunk_id for h in q.search(question, TODAY, 8, cfg)] == \
               [h.chunk.chunk_id for h in m.search(question, TODAY, 8, cfg)]
    # doc_id filter is applied server-side
    assert q.best_chunk_in_doc("hotel cap per night", "hr-travel-policy").doc_id == "hr-travel-policy"


def test_upload_embeds_only_the_new_document(qdrant_settings, tmp_path):
    kb = KnowledgeBase(qdrant_settings).load()
    before = len(kb.snapshot.chunks)
    f = tmp_path / "bike.md"
    f.write_text("# Bike Scheme\n\n## Allowance\nThe company contributes up to $750 towards a bicycle.\n", encoding="utf-8")
    kb.add_document(f, {"doc_id": "bike-scheme", "title": "Bike Scheme"})
    stats = kb.vector_store_stats
    assert stats["embedded"] == len(kb.snapshot.chunks) - before and stats["reused"] == before
    hits = kb.search("bicycle allowance", TODAY, 3, SearchConfig(use_bm25=False))
    assert hits[0].chunk.doc_id == "bike-scheme"
    # a reload without the upload deletes its now-stale vectors
    fresh = KnowledgeBase(qdrant_settings).load()
    assert fresh.vector_store_stats["deleted"] == len(kb.snapshot.chunks) - before
