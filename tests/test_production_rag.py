from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.rag.documents import DocumentChunk
from app.rag.ingestion import load_chunks
from app.rag.production import ProductionRetriever, fuse
from scripts.benchmark_retrieval import recall


def test_pdf_pages_and_stable_ids(tmp_path):
    with pymupdf.open() as doc:
        for text in ["Warranty applies for two years.", "Battery delivery restrictions."]:
            doc.new_page().insert_text((50, 50), text)
        doc.save(tmp_path / "policy.pdf")
    chunks = load_chunks(tmp_path)
    assert [c.page for c in chunks] == [1, 2]
    assert all(c.source == "policy.pdf" for c in chunks)
    assert chunks == load_chunks(tmp_path)
    assert len({c.chunk_id for c in chunks}) == 2


def test_overlap_and_bounds(tmp_path):
    (tmp_path / "long.md").write_text("abcdefg " * 100)
    chunks = load_chunks(tmp_path, 100, 20)
    assert all(len(c.text) <= 100 for c in chunks)
    assert len(chunks) > 5
    with pytest.raises(ValueError):
        load_chunks(tmp_path, 100, 100)


def test_image_only_rejected(tmp_path):
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(tmp_path / "blank.pdf")
    with pytest.raises(ValueError, match="OCR"):
        load_chunks(tmp_path)


def test_rrf_uses_rank_even_negative_cosine_and_deduplicates():
    a, b = DocumentChunk("a", "a.md", "a"), DocumentChunk("b", "b.md", "b")
    result = fuse([[(a, -0.1), (b, -0.9)], [(b, 100), (b, 99)]])
    assert result[0][0] == b
    assert result[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rerank_and_candidate_limits():
    a, b = DocumentChunk("a", "a.md", "a"), DocumentChunk("b", "b.md", "b")
    stores, model, rerank = Mock(), Mock(), Mock()
    model.encode.return_value = np.ones((1, 3))
    stores.dense_search.return_value = [(a, 0.9), (b, 0.1)]
    stores.sparse_search.return_value = [(a, 12)]
    rerank.score.return_value = [0.1, 0.9]
    retriever = ProductionRetriever(Settings(_env_file=None), stores, model, rerank)
    results = retriever.search("query", 1)
    assert results[0].chunk == b
    assert results[0].vector_score == 0.1
    assert stores.dense_search.call_args.args[1] == 20
    stores.sparse_search.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError):
        retriever.search("query")
    assert retriever.search(" ") == []


def test_recall_multiple_labels():
    row = Mock(chunk=DocumentChunk("a", "policy.pdf", "text", 2))
    assert (
        recall([row], [{"source": "policy.pdf", "page": 2}, {"source": "policy.pdf", "page": 1}])
        == 0.5
    )


def test_local_api_lifespan_and_memory(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "local")
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        for _ in range(2):
            result = client.post(
                "/chat", json={"message": "refund inspection", "session_id": "api-test"}
            )
            assert result.status_code == 200
            assert result.json()["citations"]
        assert "previous conversation" in result.json()["answer"]
    get_settings.cache_clear()


def test_fixture_has_more_than_five_candidates():
    assert len(load_chunks(Path("data/knowledge"))) > 5


def configured_stores():
    from app.rag.stores import SearchStores

    stores = SearchStores.__new__(SearchStores)
    stores.s = Settings(_env_file=None)
    stores.dense, stores.sparse = Mock(), Mock()
    stores.sparse.cluster.health.return_value = {"status": "yellow", "timed_out": False}
    stores.dense.has_collection.return_value = True
    stores.sparse.indices.get_mapping.return_value = {
        "knowledge_v1": {
            "mappings": {
                "_meta": {
                    "complete": True,
                    "embedding_model": stores.s.embedding_model,
                    "model_max_length": stores.s.model_max_length,
                }
            }
        }
    }
    stores.sparse.count.return_value = {"count": 12}
    stores.dense.query.return_value = [{"count(*)": 12}]
    return stores


def test_readiness_rejects_partial_and_model_mismatch():
    stores = configured_stores()
    stores.check()
    stores.dense.load_collection.assert_called_once_with("knowledge_v1")
    meta = stores.sparse.indices.get_mapping.return_value["knowledge_v1"]["mappings"]["_meta"]
    meta["complete"] = False
    with pytest.raises(RuntimeError, match="Incomplete"):
        stores.check()
    meta["complete"] = True
    meta["embedding_model"] = "other-model"
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        stores.check()


def test_readiness_rejects_count_mismatch_and_unhealthy_cluster():
    stores = configured_stores()
    stores.dense.query.return_value = [{"count(*)": 11}]
    with pytest.raises(RuntimeError, match="inconsistent"):
        stores.check()
    stores.sparse.cluster.health.return_value = {"status": "red", "timed_out": True}
    with pytest.raises(RuntimeError, match="not ready"):
        stores.check()


def test_store_search_payload_and_metadata():
    stores = configured_stores()
    row = {"chunk_id": "a", "source": "p.pdf", "page": 2, "text": "policy"}
    stores.dense.search.return_value = [[{"entity": row, "distance": -0.3}]]
    result = stores.dense_search(np.ones(3), 20)
    assert result[0][0].page == 2
    assert result[0][1] == -0.3
    assert stores.dense.search.call_args.kwargs["search_params"]["metric_type"] == "COSINE"
    stores.sparse.search.return_value = {"hits": {"hits": [{"_source": row, "_score": 2.4}]}}
    assert stores.sparse_search("policy", 20)[0][1] == 2.4
    assert stores.sparse.search.call_args.kwargs["query"] == {"match": {"text": "policy"}}
