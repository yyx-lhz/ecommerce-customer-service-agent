"""Run with python -m scripts.ingest. Each version is immutable once built."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from app.core.config import Settings
from app.rag.ingestion import load_chunks
from app.rag.models import BGEEmbedding
from app.rag.stores import SearchStores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-dir", type=Path, default=Path("data/knowledge"))
    args = parser.parse_args()
    s = Settings()
    chunks = load_chunks(args.knowledge_dir, s.chunk_size, s.chunk_overlap)
    model = BGEEmbedding(s)
    stores = SearchStores(s)
    try:
        stores.create(model.dimensions)
        for start in range(0, len(chunks), s.embedding_batch_size):
            batch = chunks[start : start + s.embedding_batch_size]
            stores.insert(batch, model.encode([c.text for c in batch]))
        stores.finish()
        stores.check()
        manifest = {
            "version": s.rag_index_version,
            "chunks": len(chunks),
            "embedding_model": s.embedding_model,
            "dimensions": model.dimensions,
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "corpus_sha256": hashlib.sha256(
                json.dumps([asdict(c) for c in chunks], sort_keys=True).encode()
            ).hexdigest(),
        }
        Path("reports").mkdir(exist_ok=True)
        Path(f"reports/ingestion_{s.rag_index_version}.json").write_text(
            json.dumps(manifest, indent=2)
        )
        print(json.dumps(manifest, indent=2))
    finally:
        stores.close()


if __name__ == "__main__":
    main()
