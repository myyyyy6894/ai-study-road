"""Day5：Top-K 对答案质量与 token 成本的影响。"""

import asyncio

# 允许直接运行：uv run python examples/xx.py（把项目根加入模块搜索路径）
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from app.rag.pipeline import answer

QUESTION = "Q-learning 的更新公式是什么？"


async def main() -> None:
    for k in (2, 4, 8):
        result = await answer(QUESTION, top_k=k)
        usage = result["usage"]
        scores = [c["score"] for c in result["citations"]]

        print(f"\n===== top_k={k} =====")
        print(f"检索到 {result['retrieved']} 条")
        print(f"分数：{scores}")
        print(f"输入 {usage['prompt_tokens']} token / 输出 {usage['completion_tokens']} token")
        # 只打印前 120 字，方便横向对比
        print(f"回答：{result['answer'][:120].replace(chr(10), ' ')}")


if __name__ == "__main__":
    asyncio.run(main())
