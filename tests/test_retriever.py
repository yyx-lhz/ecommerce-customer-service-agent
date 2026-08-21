from pathlib import Path

from app.rag.documents import load_markdown_chunks
from app.rag.hybrid import HybridRetriever


def test_bm25_and_vector_fusion_finds_logistics_policy():
    chunks = load_markdown_chunks(Path("data/knowledge"))
    retriever = HybridRetriever(chunks)

    results = retriever.search("DHL customs clearance tracking delay", top_k=3)

    assert results
    assert any(result.chunk.source == "logistics_policy.md" for result in results)
