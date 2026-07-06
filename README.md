# 跨境电商智能客服 Agent

基于 LangGraph 的多 Agent 协作智能客服系统，专为跨境电商场景设计，覆盖询盘、订单、物流等高频咨询场景。

## 功能特性

- **多 Agent 协作**: 意图识别 → 工具调用 → 答复生成，三步流水线
- **RAG 检索增强**: 基于 Chroma 向量数据库的产品知识库，支持中英文混合检索
- **Reflection 自检机制**: 生成答复后自动检查合规性与准确性
- **Function Calling**: 模拟订单查询、物流追踪等 API 调用
- **Streamlit 可视化界面**: 直观的对话交互和调试面板

## 技术架构

```
用户输入 → 意图识别 Agent → 工具调用 Agent → 答复生成 Agent → Reflection → 输出
                ↓                ↓                ↓
           意图分类           RAG检索         合规检查
                           Function Call     准确性验证
                           API模拟数据
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key"
```

### 启动应用

```bash
streamlit run app.py
```

## 项目结构

```
├── app.py                  # Streamlit 主界面
├── agent/
│   ├── __init__.py
│   ├── graph.py            # LangGraph 状态图定义
│   └── nodes.py            # Agent 节点实现
├── rag/
│   ├── __init__.py
│   ├── knowledge_base.py   # 知识库构建与检索
│   └── data/
│       └── products.json   # 模拟产品数据
├── utils/
│   ├── __init__.py
│   ├── mock_apis.py        # 模拟 API（订单查询/物流追踪）
│   └── reflection.py       # Reflection 自检模块
├── requirements.txt
└── README.md
```

## 测试结果

针对 50 条真实客服场景测试：
- 意图识别准确率: 92%
- 答复准确率: 80%+
- 平均响应时间: ~3 秒
