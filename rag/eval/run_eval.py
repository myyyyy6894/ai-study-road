"""Day7 评估：检索指标 + 生成质量 + 拒答准确率。

用法：
    uv run python -m eval.run_eval                      # 默认阈值 0.5
    uv run python -m eval.run_eval --min-score 0.45
    uv run python -m eval.run_eval --min-score 0.4 --top-k 6
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from app.llm import chat
from app.prompts import RAG_NO_ANSWER
from app.rag.pipeline import answer, get_store
from app.retrieval.embedder import embed_query

QA_PATH = Path("eval/qa_set.jsonl")

JUDGE_PROMPT = """你是严格的评估员，判断模型回答是否回答了问题。

问题：{question}
标准答案：{gold}
模型回答：{model_answer}

评分标准：
2 = 正确，关键事实与标准答案一致
1 = 部分正确，遗漏关键信息或含少量不准确内容
0 = 错误、答非所问、或编造了标准答案里没有的事实

只输出一个数字（0、1 或 2），不要任何解释。"""


def load_qa() -> list[dict]:
    lines = QA_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def find_gold_chunk_ids(gold_answer: str, top_n: int = 3) -> set[str]:
    """找出"可能包含答案"的片段集合。

    做法：用标准答案去检索，取前 top_n。
    为什么取 3 而不是 1：同一份标准答案往往分散在相邻几个块里
    （比如"悬崖数量"和"悬崖位置"可能被切开），只取 Top-1 会误判 Recall。
    """
    results = get_store().search(embed_query(gold_answer), top_k=top_n)
    return {chunk["chunk_id"] for chunk, _ in results}


async def judge(question: str, gold: str, model_answer: str) -> int:
    resp = await chat(
        [
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question, gold=gold, model_answer=model_answer
                ),
            }
        ],
        temperature=0.0,
        max_tokens=512,
    )
    text = (resp.choices[0].message.content or "").strip()
    for ch in text:
        if ch in "012":
            return int(ch)
    return 0


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--tag", type=str, default="baseline")
    args = parser.parse_args()

    qa = load_qa()
    get_store()  # 预热：提前加载索引，避免首次检索的磁盘读取混入延迟统计

    recall_hits = 0
    in_scope_total = 0
    scores: list[int] = []
    refusal_correct = 0
    out_scope_total = 0
    latencies: list[float] = []
    total_tokens = 0
    rows: list[dict] = []

    for item in qa:
        started = time.perf_counter()
        result = await answer(item["question"], top_k=args.top_k, min_score=args.min_score)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        if "usage" in result:
            total_tokens += result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]

        if item["type"] == "out_of_scope":
            out_scope_total += 1
            refused = RAG_NO_ANSWER in result["answer"] or "无法回答" in result["answer"]
            refusal_correct += int(refused)
            rows.append(
                {
                    "id": item["id"],
                    "type": "库外",
                    "question": item["question"],
                    "answer": result["answer"][:60],
                    "score": "拒答" if refused else "未拒答",
                    "recall": "-",
                    "latency": elapsed,
                }
            )
            print(f"[{item['id']:2d}] 库外  {'拒答' if refused else '❌未拒答'}  {elapsed:.1f}s")
            continue

        in_scope_total += 1
        gold_ids = find_gold_chunk_ids(item["gold_answer"])
        retrieved_ids = {c["chunk_id"] for c in result["citations"]}
        hit = bool(gold_ids & retrieved_ids)
        recall_hits += int(hit)

        score = await judge(item["question"], item["gold_answer"], result["answer"])
        scores.append(score)

        rows.append(
            {
                "id": item["id"],
                "type": "库内",
                "question": item["question"],
                "answer": result["answer"][:60],
                "score": score,
                "recall": "✅" if hit else "❌",
                "latency": elapsed,
            }
        )
        print(
            f"[{item['id']:2d}] 库内  Recall={'✅' if hit else '❌'}  评分={score}  {elapsed:.1f}s"
        )

    recall = recall_hits / in_scope_total if in_scope_total else 0
    accuracy = sum(1 for s in scores if s >= 2) / len(scores) if scores else 0
    partial = sum(1 for s in scores if s >= 1) / len(scores) if scores else 0
    refusal = refusal_correct / out_scope_total if out_scope_total else 0
    avg_latency = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]

    report = f"""# RAG 评估报告：{args.tag}

- 评估集：{len(qa)} 条（库内 {in_scope_total}，库外 {out_scope_total}）
- Top-K：{args.top_k}
- 阈值：{args.min_score}

## 核心指标

| 指标 | 数值 |
|---|---|
| Recall@{args.top_k} | **{recall:.1%}** |
| 完全正确率（评分=2） | **{accuracy:.1%}** |
| 可接受率（评分≥1） | **{partial:.1%}** |
| 库外拒答准确率 | **{refusal:.1%}** |
| 平均延迟 | {avg_latency:.2f}s |
| P95 延迟 | {p95:.2f}s |
| 总 token | {total_tokens} |

## 逐题明细

| ID | 类型 | 问题 | Recall | 评分 | 延迟 |
|---|---|---|---|---|---|
"""
    for r in rows:
        report += f"| {r['id']} | {r['type']} | {r['question']} | {r['recall']} | {r['score']} | {r['latency']:.1f}s |\n"

    out = Path(f"eval/report_{args.tag}.md")
    out.write_text(report, encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"[{args.tag}] Recall@{args.top_k}: {recall:.1%}")
    print(f"完全正确率: {accuracy:.1%}   可接受率: {partial:.1%}")
    print(f"库外拒答率: {refusal:.1%}")
    print(f"平均延迟: {avg_latency:.2f}s   P95: {p95:.2f}s")
    print(f"总 token: {total_tokens}")
    print(f"报告: {out}")


if __name__ == "__main__":
    asyncio.run(main())
