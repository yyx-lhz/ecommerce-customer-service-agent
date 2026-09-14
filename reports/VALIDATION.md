# 实际验证记录（2026-09-14）

验证代码提交：`1b74381fc4d167698dc3e5754c5a2c79738542ba`。

[GitHub Actions 完整运行](https://github.com/yyx-lhz/ecommerce-customer-service-agent/actions/runs/34821656203)：unit、image、integration 三个任务全部成功。报告提交仅补充证据，不改变被验证的运行代码。

## 已执行的检查

- macOS / Python 3.13 与 GitHub Ubuntu / Python 3.11：15 项单元测试通过。
- Ruff 静态检查、compileall、Compose 配置检查通过；本地 pip check 未发现损坏的依赖。
- Dockerfile 实际构建成功，构建后的镜像可导入应用和 Milvus SDK。
- Compose 实际启动 Redis、Elasticsearch、etcd、MinIO、Milvus，健康检查通过。
- 真实 PyMuPDF 解析、BGE-M3 推理、Milvus 与 Elasticsearch 写入成功；两库计数及完成标记校验成功。
- 真实 BGE-M3 + Milvus dense + ES BM25 + RRF + BGE-Reranker 完成离线检索评测。
- FastAPI TestClient 通过真实数据库及模型调用 `/ready`、`/chat`，保修查询返回 `sample_policies.pdf` 第 1 页。原始响应见 `smoke_production.json`。

真实链路验证采用 **CI 宿主机 Python + Compose 数据库服务**。完整 API 容器的联网启动未另行执行；容器构建与导入已验证。本机模型和镜像下载较慢，完整模型验证转到上述 CI 完成。

## 实际输出

| 后端 | 查询数 | 最终 source/page Recall@5 |
| --- | ---: | ---: |
| local Hash / 内存 BM25 / RRF | 12 | 1.0 |
| 真实 BGE / Milvus / ES / RRF / Reranker | 12 | 1.0 |

- 原始报告：`retrieval_local.json`、`retrieval_production.json`，保留逐题来源、页码、分数、耗时和代码/数据版本。
- 实际语料：4 份 Markdown + 1 份 8 页合成 PDF，共 12 个片段。PDF 相关性按 source/page，Markdown 按 source 标注；不是 chunk 级评价。
- 入库记录：`ingestion_ci.json`。真实模型 `BAAI/bge-m3` 输出 1024 维，默认字符窗口 1000、重叠 150。
- 模型精排：`BAAI/bge-reranker-v2-m3`。
- 报告中的 `working_tree_dirty` 检查排除 `reports/`，避免运行时生成证据文件被误认为源代码改动。

这是小型合成回归集。两种后端都满分，**不能证明新链路比本地基线质量更好**；没有业务线上指标、吞吐压测或延迟 SLA。逐题耗时包含首次运行影响，不作为性能承诺。

## 可准确写进简历

“基于 PyMuPDF 实现 PDF 文档解析与带页码的分块入库，使用 BGE-M3 生成文本向量，结合 Milvus Dense Retrieval 与 Elasticsearch BM25 双路召回，通过 RRF 融合及 BGE-Reranker 精排构建混合检索流程，并接入 FastAPI 客服工作流，完成真实模型和数据库的端到端验证。”

如需写数字，只能限定为“在 12 条合成查询的 source/page 级回归集上，实测 Recall@5 为 1.0”。不建议把这个小样本数字作为核心业务成绩。

当前不可据此声称：真实业务准确率提升、生产规模验证、完整 LLM 回答生成、真实订单接口接入、OCR 或复杂表格解析。`app/` 的回答仍采用模板拼接，业务工具仍为 mock，旧规则 Reflection 值不等于独立事实正确率。
