"""Day 1 练习：LLM API 单轮 / 多轮对话。

运行：
    uv run python -m scripts.day1_hello_llm            # 单轮 + 参数实验
    uv run python -m scripts.day1_hello_llm multi      # 多轮交互对话
    uv run python -m scripts.day1_hello_llm inflate    # 观察 messages 如何膨胀
"""

import asyncio
import sys

from app.llm import chat

SYSTEM = "你是一个简洁的中文技术助手，回答控制在 3 句话以内。"


def show(resp, title: str = "") -> None:
    """把一次响应里我们关心的东西打印出来。"""
    if title:
        print(f"\n===== {title} =====")
    choice = resp.choices[0]
    usage = resp.usage
    print(f"[回答] {choice.message.content}")
    print(f"[finish_reason] {choice.finish_reason}")
    print(
        f"[usage] 输入 {usage.prompt_tokens} + 输出 {usage.completion_tokens} "
        f"= {usage.total_tokens} tokens"
    )
    # DeepSeek 特有的上下文缓存统计：命中部分更便宜
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if hit is not None:
        print(f"[缓存] 命中 {hit} tokens / 未命中 {usage.prompt_cache_miss_tokens} tokens")


async def single_turn() -> None:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "用三句话解释什么是 RAG，以及它解决了大模型的什么问题。"},
    ]
    show(await chat(messages), "单轮调用")

    # 实验：max_tokens 设得很小，观察 finish_reason 变成 length（回答被硬截断）
    show(await chat(messages, max_tokens=20), "把 max_tokens 限制为 20")

    # 实验：temperature 对同一问题的影响（连续两次，看措辞是否变化）
    show(await chat(messages, temperature=0.0), "temperature=0")
    show(await chat(messages, temperature=1.4), "temperature=1.4")


async def multi_turn() -> None:
    """核心演示：API 本身无状态，历史全靠我们自己维护。"""
    messages = [{"role": "system", "content": SYSTEM}]
    print("多轮对话（输入 exit 退出）。试试：先告诉它你的名字，下一轮再问它。")

    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        resp = await chat(messages)
        answer = resp.choices[0].message.content
        print(f"AI > {answer}")

        # ★ 关键一步：把模型的回复也追加进历史，下一轮才"记得"
        messages.append({"role": "assistant", "content": answer})

        chars = sum(len(m["content"]) for m in messages)
        print(f"   （当前 messages 共 {len(messages)} 条，{chars} 字符）")

    print(f"\n[最终 messages] {len(messages)} 条")
    for m in messages:
        print(f"  {m['role']:9s}: {m['content'][:40]}")


async def inflate() -> None:
    """证明'记忆'的代价：每轮都重发全部历史，输入 token 线性增长。"""
    messages = [{"role": "system", "content": SYSTEM}]
    for i in range(1, 6):
        messages.append({"role": "user", "content": f"请重复这句话，一个字都不要改：第{i}轮测试。"})
        resp = await chat(messages)
        messages.append({"role": "assistant", "content": resp.choices[0].message.content})
        print(f"第 {i} 轮：messages {len(messages)} 条，输入 token = {resp.usage.prompt_tokens}")


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    match mode:
        case "single":
            await single_turn()
        case "multi":
            await multi_turn()
        case "inflate":
            await inflate()
        case _:
            print(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
