# ai-study

从零手写 AI 应用的学习仓库。每个子项目都是独立可运行的服务，**不使用封装框架**，核心逻辑全部自己实现，并附带实测数据说明参数选择的依据。

## 项目

| 项目 | 状态 | 内容 |
|---|---|---|
| [`rag/`](rag/) | ✅ 完成 | 知识库问答：文档解析、切片、向量检索、带引用回答、拒答 |
| [`agent/`](agent/) | 🚧 计划中 | Tool Calling、Agent 执行循环、多工具编排 |

## rag/ 能做什么

上传 PDF/Markdown/TXT → 自动切片向量化 → 自然语言提问 → 返回带引用的回答；知识库里没有答案时明确拒答。

**实测指标**（25 条评估集）：Recall@8 = 90.5%，完全正确率 95.2%，库外拒答率 100%，平均延迟 1.2s。

**技术栈**：FastAPI · DeepSeek · BGE-M3（Ollama）· numpy 向量检索 · 原生 HTML + SSE

```bash
cd rag
uv sync
cp .env.example .env    # 填入 DEEPSEEK_API_KEY
uv run python scripts/rebuild_index.py
uv run uvicorn app.main:app --port 8000
```

打开 http://127.0.0.1:8000/ 即可使用。详见 [`rag/README.md`](rag/README.md)。

## 文档

- [`rag/docs/WALKTHROUGH.md`](rag/docs/WALKTHROUGH.md) —— 3418 行全链路讲解，10 个部分：
  - Part 1 全景图与两条链路
  - Part 2 从按下发送键到进入函数（HTTP / 路由 / Pydantic / async）
  - Part 3 会话与 query 改写
  - Part 4 文档怎么变成向量（解析 / 页眉清洗 / 切片 / embedding）
  - Part 5 向量存储与 Top-K 检索
  - Part 6 检索结果怎么变成答案（阈值 / Prompt / 引用映射）
  - Part 7 调用 LLM 的细节（SDK / 代理坑 / 流式生成器）
  - Part 8 SSE 与前端渲染（半条消息 / XSS）
  - Part 9 索引的增删改查
  - Part 10 评估、优化与面试速查表

每个部分末尾都有自检问题。

## 设计原则

1. **不套框架**：LangChain 会把"检索"变成三行黑盒，调优时你不知道该改哪。手写一遍才知道 `chunk_size=400` 和阈值 `0.45` 是怎么来的。
2. **每个参数都有实验**：从 chunk 大小到 Top-K，都跑过对照实验并把数据写进 README。
3. **每个设计都有被否决的备选**：为什么不用向量数据库、为什么不用 WebSocket，文档里都有答案。

## License

MIT
