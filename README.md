# 🛍️ 跨境电商智能客服 Agent

> 基于 LangGraph 的多 Agent 协作智能客服系统 | FastAPI + Docker + 多轮对话记忆 + RAG + Reflection

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 核心特性

| 特性 | 说明 | 技术实现 |
|------|------|---------|
| 🧠 **多 Agent 协作** | 意图识别→RAG检索→工具调用→答复生成→自检，5 步流水线 | LangGraph StateGraph |
| 🔍 **RAG 检索增强** | 产品知识库中英文混合检索，FAQ 精确匹配 | Chroma + OpenAI Embedding |
| 🛠️ **Function Calling** | 自动路由到订单查询/物流追踪/库存检查/产品信息/退货政策 | OpenAI Tool Calling |
| 🪞 **Reflection 自检** | 生成答复后二次审核，检查合规性、准确性、语调，自动修正 | LLM-as-Judge |
| 💬 **多轮对话记忆** | Session 级别的对话历史管理，支持上下文连续对话 | In-Memory Store |
| 🐳 **Docker 部署** | 一键 `docker-compose up`，开箱即用 | Docker + docker-compose |
| 🚀 **FastAPI 后端** | REST API + Swagger 文档，生产级服务化 | FastAPI + Uvicorn |
| 📊 **评估体系** | 50 条测试用例，意图准确率 90%+，响应 < 3s | 自建 Eval 框架 |
| 🎨 **Streamlit 调试面板** | 可视化对话 + 全链路 Trace 展示 | Streamlit |

## 🏗️ 项目架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Web UI (Streamlit / API)                 │
├─────────────────────────────────────────────────────────────┤
│                    LangGraph Agent Pipeline                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ ① Intent │→│ ② RAG    │→│ ③ Tool   │→│ ④ Response│   │
│  │ Classify │  │ Retrieve │  │ Execute  │  │ Generate  │   │
│  └──────────┘  └──────────┘  └────┬─────┘  └─────┬─────┘   │
│                                    │               │        │
│                              ┌─────▼──────┐  ┌─────▼──────┐ │
│                              │ Mock APIs  │  │ ⑤ Reflect  │ │
│                              │(Order/物流)│  │  Self-Check│ │
│                              └────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────┤
│         RAG Layer            │        Memory Layer           │
│  ┌───────────────────┐       │  ┌────────────────────┐      │
│  │ Chroma Vector DB  │       │  │ Session Store       │      │
│  │ + OpenAI Embedding│       │  │ (Multi-turn Dialog) │      │
│  └───────────────────┘       │  └────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/yyx-lhz/ecommerce-customer-service-agent.git
cd ecommerce-customer-service-agent
pip install -r requirements.txt
```

### 2. 设置 API Key

```bash
export OPENAI_API_KEY="sk-your-key"
```

### 3. 启动方式

**方式 A: FastAPI (推荐用于生产)**
```bash
uvicorn api:app --reload --port 8000
# 访问 http://localhost:8000/docs 查看 Swagger API 文档
```

**方式 B: Streamlit (推荐用于演示/调试)**
```bash
streamlit run app.py
# 访问 http://localhost:8501
```

**方式 C: Docker**
```bash
docker-compose up -d
# API 服务在 http://localhost:8000
```

### 4. API 调用示例

```bash
# 单轮对话
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我查一下订单ORD-1001"}'

# 多轮对话
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "蓝牙耳机多少钱？", "session_id": "user_001"}'

curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "那个能防水吗？", "session_id": "user_001"}'
```

## 📂 项目结构

```
.
├── api.py                  # FastAPI 后端 + Swagger 文档
├── app.py                  # Streamlit 调试界面
├── Dockerfile              # Docker 镜像
├── docker-compose.yml      # 一键部署
├── requirements.txt        # Python 依赖
├── agent/
│   ├── graph.py            # LangGraph 状态图编排 + 多轮记忆
│   └── nodes.py            # 5 个 Agent 节点实现
├── rag/
│   ├── knowledge_base.py   # Chroma 向量知识库
│   └── data/products.json  # 5 个模拟跨境产品数据
├── utils/
│   ├── mock_apis.py        # 模拟 API (订单/物流/库存)
│   └── reflection.py       # Reflection 自检模块
└── eval/
    └── run_eval.py         # 50 条测试用例评估框架
```

## 📊 评估结果

50 条真实客服场景测试覆盖 6 种意图：

```
Intent breakdown:
  产品咨询   90.0% [█████████░] (9/10)
  订单状态   90.0% [█████████░] (9/10)
  物流      100.0% [██████████] (10/10)
  退换货    90.0% [█████████░] (9/10)
  投诉      80.0% [████████░░] (4/5)
  其他      80.0% [████████░░] (4/5)

Overall: 90.0% accuracy | Avg latency: ~3s
```

## 🔧 技术栈

- **Agent 框架**: LangGraph (状态图编排)
- **LLM**: OpenAI gpt-4o-mini (意图分类 + 答复生成 + 自检)
- **Embedding**: OpenAI text-embedding-3-small
- **向量数据库**: Chroma (持久化存储)
- **后端**: FastAPI + Uvicorn
- **前端**: Streamlit
- **部署**: Docker + docker-compose
- **可观测性**: LangSmith (可选)
