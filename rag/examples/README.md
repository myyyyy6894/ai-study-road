# 示例脚本

按学习顺序排列。每个脚本都可以独立运行，对应 RAG 链路中的一个知识点。

运行前请确保：

- 已完成根目录的安装（`uv sync`）
- `.env` 里填好了 `DEEPSEEK_API_KEY`
- Ollama 已启动且拉取了 `bge-m3`（从 06 开始需要）

```bash
# 在 rag/ 目录下
uv run python examples/01_llm_minimal.py
```

## 索引

| 脚本 | 知识点 | 需要 Ollama |
|---|---|---|
| `01_llm_minimal.py` | 最小 LLM 调用：客户端、messages、取回答 | 否 |
| `02_response_structure.py` | 响应结构：finish_reason、usage、reasoning_content | 否 |
| `03_multi_turn.py` | 多轮对话：历史必须自己维护，对比保留/丢弃 assistant | 否 |
| `04_prompt_comparison.py` | Prompt 三档对比：模糊 / 明确 / few-shot | 否 |
| `05_structured_output.py` | JSON mode、容错解析、Pydantic 校验 | 否 |
| `06_embedding_similarity.py` | 语义相似度矩阵，同义句 vs 无关句 | 是 |
| `07_build_index.py` | 解析 → 切片 → 向量化 → 落盘 | 是 |
| `08_rag_contrast.py` | 五组对照：有无检索、库内/库外/半相关 | 是 + LLM |
| `09_topk_cost.py` | Top-K 对答案质量与 token 成本的影响 | 是 + LLM |
| `10_query_rewrite.py` | 多轮追问的 query 改写，改写前后检索对比 | 是 + LLM |

## 从示例到服务

示例是"一次性的验证脚本"，服务是"长期运行的系统"。两者的对应关系：

| 示例 | 服务中的位置 |
|---|---|
| 01~03 | `app/llm.py` |
| 04~05 | `app/prompts.py` |
| 06 | `app/retrieval/embedder.py` |
| 07 | `app/ingest/*` + `scripts/rebuild_index.py` |
| 08~09 | `app/rag/pipeline.py` + `eval/run_eval.py` |
| 10 | `app/rag/memory.py` |

建议顺序：先跑示例理解单点，再看服务代码理解如何组合。
