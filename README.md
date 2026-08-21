# Cross-border Ecommerce Customer Service Agent

An AI application engineering project for a cross-border ecommerce customer service scenario. It demonstrates intent routing, planning, hybrid RAG retrieval, business tool calling, conversation memory, reflection checks, and an evaluation pipeline.

## Capabilities

- FastAPI service with `/chat` and `/health` endpoints.
- Agent workflow: Intent Router -> Planner -> Hybrid Retriever -> Tool Executor -> Reflection.
- Business tools for order lookup, logistics tracking, inventory query, and return case creation.
- Hybrid retrieval with local vector embedding, BM25 keyword retrieval, and RRF fusion.
- Conversation memory with in-memory default and optional Redis backend.
- Docker Compose deployment for API, Redis, Elasticsearch, and Milvus standalone dependencies.
- Offline evaluation for Intent Accuracy, Tool Accuracy, Recall@5, and Faithfulness.

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Chat request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"Can you check my order OD1002?"}'
```

## Docker Deployment

```bash
cp .env.example .env
docker compose up --build
```

For a cloud server, install Docker and Docker Compose, copy the repository to the server, configure `.env`, then run the command above. The compose file uses persistent volumes for Redis, Elasticsearch, Milvus, etcd, and MinIO. The API is exposed on port `8000`.

Recommended production hardening:

- Put Nginx/Caddy in front of the API and terminate HTTPS there.
- Set `restart: unless-stopped` for long-running services.
- Keep `.env` out of Git and inject secrets through server-side environment variables.
- Add health checks and log shipping.
- Back up vector/search/business data volumes.

## Evaluation

Run the deterministic offline evaluation:

```bash
python scripts/evaluate.py
```

Expected output shape:

```json
{
  "case_count": 6,
  "intent_accuracy": 1.0,
  "tool_accuracy": 1.0,
  "recall_at_5": 1.0,
  "faithfulness": 1.0
}
```

Metrics:

- `intent_accuracy`: whether the router selected the expected intent.
- `tool_accuracy`: whether the correct business tool was called or no tool was called when expected.
- `recall_at_5`: whether the expected policy document appears in top-5 citations.
- `faithfulness`: whether the reflection module sees enough retrieval/tool evidence for the answer.

To expand the benchmark, add cases in `data/eval/customer_service_cases.json`. For resume-grade reporting, keep the dataset fixed, record the commit SHA, and report metrics from the same command.

## Tests

```bash
pytest
```

## Architecture

```text
Client
  -> FastAPI /chat
  -> Intent Router
  -> Planner
  -> Hybrid Retriever: local vector + BM25 + RRF
  -> Tool Executor: order/logistics/inventory/returns
  -> Response Composer
  -> Reflection: grounding and tool evidence checks
  -> Redis or in-memory conversation memory
```

## Notes

The default implementation is fully offline so reviewers can run it without API keys. The optional `integrations` dependency group and `.env` settings prepare the project for OpenAI embeddings, Milvus vector storage, and Elasticsearch indexing.
