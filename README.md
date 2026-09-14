# Ecommerce Customer Service Agent

以 `app/` FastAPI 服务为主的客服检索工程。支持两种**显式**运行模式：

| 模式 | 实际行为 |
| --- | --- |
| `RAG_BACKEND=local`（默认，兼容旧演示） | Markdown 段落、本地 Hash 向量、内存 BM25、RRF；无真实语义模型或数据库 |
| `RAG_BACKEND=production` | PDF/Markdown 离线入库、BGE-M3、Milvus、Elasticsearch BM25、RRF、BGE-Reranker |

`production` 表示真实后端实现，不代表已完成生产容量或安全认证。模型/服务缺失会报错，不自动降级为 local。

## 实际链路与关键代码

```text
data/knowledge PDF / Markdown
  -> PyMuPDF 按页提取 PDF 文本（Markdown 页码为 0）
  -> 清理空白、边界切分、重叠窗口、稳定 chunk_id
  -> BAAI/bge-m3 归一化 dense embedding
  -> Milvus HNSW / COSINE（向量 + 原文 + source/page/chunk_id）
  -> Elasticsearch 倒排索引 / BM25（相同 chunk_id）

/chat -> 现有 Agent Workflow
  -> BGE-M3 query embedding
  -> 并行 Milvus dense Top-20 + Elasticsearch BM25 Top-20
  -> RRF: sum(1 / (60 + rank))，按 chunk_id 去重
  -> Top-20 BAAI/bge-reranker-v2-m3 CrossEncoder
  -> 重排 Top-5 -> 引用和现有回答拼接器
```

- `app/rag/ingestion.py`：递归解析 PDF/Markdown。默认 1000 字符窗口、150 字符重叠，优先换行/句末边界，不跨 PDF 页。空白/扫描页明确报 OCR 错误；未实现 OCR、复杂表格或阅读顺序恢复。
- `app/rag/models.py`：通过 SentenceTransformers 加载真实 BGE-M3（默认 1024 维）和 BGE-Reranker。模型权重首次下载，之后缓存；无付费推理 API。默认 CPU，编码批次为 8。
- `app/rag/stores.py`：Milvus 强一致性、HNSW `M=16, efConstruction=200`、COSINE；ES 显式 BM25 `k1=1.2, b=0.75` 和 standard analyzer。中文未配置专用词法分析器。稀疏一路使用 ES BM25，不使用 BGE-M3 的 learned sparse 输出。
- `app/rag/production.py`：双路召回、RRF、精排。保留负 cosine 候选，RRF 不混加原始分数。返回的 `score` 是 CrossEncoder 分数；另存 dense/BM25/RRF 分数用于调试。
- `app/main.py`：启动时加载一个 Agent/模型实例，关闭时释放数据库连接。`/health` 是存活检查，`/ready` 验证索引、完整标记、模型配置及两库计数。启动前必须完成 ingestion。
- `app/agent/workflow.py`：原意图路由、工具调用、记忆、规则 Reflection 保留。**回答仍是模板拼接，不是 LLM 生成；业务 API 仍为 mock；Reflection 不等于经独立验证的事实正确率。**

模型用法依据 [BGE-M3 官方模型卡](https://huggingface.co/BAAI/bge-m3) 与 [BGE-Reranker 官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)。

## 本地兼容演示

建议 Python 3.11+。基础演示无需 Docker 或模型下载：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
uvicorn app.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Please check order OD1002","session_id":"demo"}'
```

## 真实后端：宿主机运行模型 + Docker 数据库

Docker Desktop 或 Linux Docker Engine 必须运行。模型权重和容器首次下载需要网络及数 GB 磁盘；内存消耗与模型、批次、容器配置有关。CPU 运行较慢。

```bash
pip install -e '.[dev,integrations]'
docker compose up -d redis elasticsearch milvus
# 确认所有服务 healthy 后入库
docker compose ps
python -m scripts.ingest
RAG_BACKEND=production uvicorn app.main:app --port 8000
```

另一个终端验证：

```bash
curl -f http://localhost:8000/ready
python -m scripts.smoke_production
python -m scripts.benchmark_retrieval --backend production --output reports/retrieval_production.json
```

环境配置见 `.env.example`。使用 `MODEL_DEVICE=cuda` 或 `mps` 前请确认运行环境支持。`MODEL_MAX_LENGTH` 默认 1024 tokens；模型会截断更长输入，调整字符 chunk 大小时应同步考虑 token 上限。`MILVUS_URI` 可覆盖 host/port；可配置 `MILVUS_TOKEN`。本 Compose 为本机开发，数据库关闭鉴权且仅绑定 localhost；连接受保护的 ES 部署需另行配置客户端认证/TLS。

## 完整 Compose 运行

```bash
cp .env.example .env
docker compose up -d redis elasticsearch milvus
docker compose --profile tools run --build --rm ingest
docker compose up --build -d api
curl -f http://localhost:8000/ready
```

模型缓存使用持久卷，知识库只读挂载；Compose 明确使用 production 和容器内部服务地址。支持 `RAG_INDEX_VERSION` 选择索引版本，其他模型/召回参数可在 Compose 的共享 `rag-env` 中设置。API 健康检查等待模型加载完成。

## 文档更新与故障恢复

每次 ingestion 必须使用新的 `RAG_INDEX_VERSION`，对应两库 `knowledge_<version>`。不覆盖既有索引，不在 `/chat` 中重新解析或写入文档。

```bash
RAG_INDEX_VERSION=v2 python -m scripts.ingest
# 成功后用同一版本启动 API/benchmark；也可写入 .env
RAG_INDEX_VERSION=v2 RAG_BACKEND=production uvicorn app.main:app
```

两库不是分布式事务：中途失败可能留下不完整版本；不会切换正在运行的 API。修复故障后用新版本重新入库，旧/失败版本由管理员确认不再使用后清理。ES 完成标记、模型名/长度校验和计数检查拦截常见不完整版本；不支持同版本并发 ingestion。改变语料、模型或切分配置需新建版本，切换后重启 API。不是零停机别名切换方案。

`reports/ingestion_<version>.json` 记录语料哈希、模型和切分参数。报告仅在入库成功后写入，Compose 的临时 ingest 容器同时输出报告到终端。

## 测试与真实评测

```bash
python -m pytest -q
ruff check app tests scripts
python -m compileall -q app scripts
docker compose config --quiet
python -m scripts.benchmark_retrieval --backend local --output reports/retrieval_local.json
# 依赖真实服务、真实权重和完成的 ingestion；任何失败都不会改跑 local
python -m scripts.benchmark_retrieval --backend production --output reports/retrieval_production.json
python -m scripts.smoke_production
```

提交了 8 页合成 PDF `data/knowledge/sample_policies.pdf` 和 4 份原 Markdown。PDF 可用 `python -m scripts.make_fixture` 重新生成。`data/eval/retrieval_cases.json` 为 12 个固定查询，PDF 按 source/page 标注，Markdown 按 source 标注；指标是每个查询在最终 Top-5 中命中的相关标注比例，再对查询取均值。不是严格 chunk 级 Recall。合成数据仅用于工程回归，不代表真实业务质量，也不足以证明混合检索优于基线。

所有指标由脚本输出，报告含数据哈希、代码提交、工作树状态、后端名称与逐题命中。实际执行情况见 `reports/VALIDATION.md`，不提供“预期满分”。旧 `scripts/evaluate.py` 保留用于规则 Agent 的 6 例回归，其 `faithfulness` 是内部规则值，不能作为独立忠实度评测。

## 旧版本兼容

`rag/` 仍是 Chroma + OpenAI embedding；`agent/`、`utils/`、`api.py`、`app.py` 为原 LangGraph/Streamlit 演示，未删除或改成新实现。在单独虚拟环境安装 `requirements.txt` 后使用原入口：

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
uvicorn api:app --port 8001
streamlit run app.py
```

本次新链路通过 `app.main:app` 运行。旧入口依赖及在线 OpenAI 行为未在本次新链路测试中验证。
