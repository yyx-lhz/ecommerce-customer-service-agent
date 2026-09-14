"""Offline retrieval evaluation; never substitutes mocks for production."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from app.core.config import Settings
from app.rag.hybrid import HybridRetriever
from app.rag.ingestion import load_chunks
from app.rag.production import ProductionRetriever


def recall(results, relevant):
    if not relevant:
        raise ValueError("Each query must have relevance labels")
    return sum(
        any(
            r.chunk.source == label["source"]
            and ("page" not in label or r.chunk.page == label["page"])
            for r in results[:5]
        )
        for label in relevant
    ) / len(relevant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "production"], default="production")
    parser.add_argument("--cases", type=Path, default=Path("data/eval/retrieval_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval.json"))
    args = parser.parse_args()
    s = Settings(rag_backend=args.backend)
    retriever = (
        ProductionRetriever(s)
        if args.backend == "production"
        else HybridRetriever(load_chunks(Path("data/knowledge"), s.chunk_size, s.chunk_overlap))
    )
    details = []
    try:
        for case in json.loads(args.cases.read_text()):
            start = time.perf_counter()
            results = retriever.search(case["query"], top_k=5)
            details.append(
                {
                    "query": case["query"],
                    "recall_at_5": recall(results, case["relevant"]),
                    "seconds": time.perf_counter() - start,
                    "hits": [
                        {
                            "source": r.chunk.source,
                            "page": r.chunk.page,
                            "chunk_id": r.chunk.chunk_id,
                            "score": r.score,
                        }
                        for r in results
                    ],
                }
            )
        if not details:
            raise ValueError("Empty benchmark")
        report = {
            "backend": args.backend,
            "case_count": len(details),
            "recall_at_5": sum(d["recall_at_5"] for d in details) / len(details),
            "label_unit": "source/page; source-only for Markdown",
            "dataset_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
            "index_version": s.rag_index_version,
            "embedding_model": s.embedding_model
            if args.backend == "production"
            else "LocalHashEmbedding",
            "reranker": s.reranker_model if args.backend == "production" else None,
            "details": details,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    finally:
        close = getattr(retriever, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
