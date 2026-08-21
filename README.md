# Cross-border Ecommerce Customer Service Agent

跨境电商智能客服 Agent 项目，面向商品咨询、订单查询、物流追踪、库存查询、退换货咨询等客服场景。项目包含两套实现痕迹：

- `app/`: 新增的可复现 FastAPI Agent 服务，默认离线可跑，包含 RAG、工具调用、记忆、Reflection 和评测脚本。
- `agent/`, `rag/`, `utils/`, `api.py`, `app.py`: 原仓库的 LangGraph / Streamlit 演示版本，保留用于展示早期实现思路。

## Capabilities

- FastAPI service with `/chat` and `/health` endpoints.
- Agent workflow: Intent Router -> Planner -> Hybrid Retriever -> Tool Executor -> Reflection.
- Customer Service Case workflow with `CREATED`, `ANALYZING`, `WAITING_FOR_INFORMATION`, `EXECUTING`, `COMPLETED`, and `FAILED` states.
- Business tools for order lookup, logistics tracking, inventory query, and return case creation.
- Tool reliability layer with schema validation, parameter checks, timeout handling, retries, error handling, and idempotency for side-effect tools.
- Hybrid retrieval with local vector embedding, BM25 keyword retrieval, and RRF fusion.
- Structured customer memory for profile, order context, current case, previous actions, and important business facts.
- Agent trace observability with `trace_id`, selected tools, arguments, results, retrieved documents, and final response.
- Docker Compose deployment for API, Redis, Elasticsearch, and Milvus standalone dependencies.
- Offline evaluation for Intent Accuracy, Tool Selection Accuracy, Tool Argument Accuracy, Task Completion Rate, Recall@5, and Faithfulness.

## Architecture

```text
Client
  -> FastAPI /chat
  -> Intent Router
  -> Planner
  -> Case Workflow
  -> Hybrid Retriever: local vector + BM25 + RRF
  -> Reliable Tool Executor: validation + retry + timeout + idempotency
  -> Response Composer
  -> Reflection: grounding and tool evidence checks
  -> Structured memory and trace recording
```

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

The response includes `trace_id`, `case`, `memory`, `intent`, `citations`, `tool_calls`, `reflection`, `trace`, and structured `observability` events, so reviewers can inspect the full reasoning and execution path.

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

Current expected output:

```json
{
  "case_count": 9,
  "intent_accuracy": 1.0,
  "tool_selection_accuracy": 1.0,
  "tool_argument_accuracy": 1.0,
  "task_completion_rate": 1.0,
  "recall_at_5": 1.0,
  "faithfulness": 1.0
}
```

Metrics:

- `intent_accuracy`: whether the router selected the expected intent.
- `tool_selection_accuracy`: whether the correct business tool was called or no tool was called when expected.
- `tool_argument_accuracy`: whether tool arguments such as `order_id` and `tracking_no` match the expected business context.
- `task_completion_rate`: whether the Customer Service Case reached the expected status.
- `recall_at_5`: whether the expected policy document appears in top-5 citations.
- `faithfulness`: whether the reflection module sees enough retrieval/tool evidence for the answer.

The benchmark in `data/eval/customer_service_cases.json` covers product inquiry, order query, logistics exception, refund request, return request, and multi-turn complex questions. To run regression tests after changing prompt, workflow, or retriever logic, keep the dataset fixed and run `python scripts/evaluate.py`.

## Tests

```bash
pytest
ruff check app tests scripts
```

Verified locally:

```text
11 passed
intent_accuracy: 1.0
tool_selection_accuracy: 1.0
tool_argument_accuracy: 1.0
task_completion_rate: 1.0
recall_at_5: 1.0
faithfulness: 1.0
```

## Legacy Demo

The previous LangGraph/Streamlit implementation is still available:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-your-key"
uvicorn api:app --reload --port 8001
streamlit run app.py
```

Use the new `app.main:app` service for deterministic local review and evaluation, and the legacy demo when you want to show LLM-powered interactive behavior.
