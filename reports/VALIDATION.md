# Validation evidence

Local macOS Python 3.13: 12 unit tests passed; Ruff and compileall passed; Compose configuration passed. Local FastAPI lifecycle and /chat tested with TestClient.

`retrieval_local.json` is an actual local-hash baseline run: 12 synthetic queries, source/page Recall@5 = 1.0. It is not a BGE/Milvus/Elasticsearch result.

Real backend validation is in progress; no production metric is claimed here yet. CI runs the same ingestion, benchmark and FastAPI smoke commands with real model weights and services and uploads their output as artifacts.
