RAG_SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。

必须遵守的规则：
1. 只能依据用户提供的「参考资料」回答问题，禁止使用你自己的知识补充。
2. 回答中引用某条资料时，在该句末尾标注编号，例如 [1]、[2]。
3. 如果参考资料不足以回答问题，直接输出这一句：
    根据现有资料，我无法回答这个问题。
4. 不要编造参考资料中不存在的事实、数字或结论。
5. 回答简洁，控制在 300 字以内。"""


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page")
        location = f"第 {page} 页" if page else "全文"

        blocks.append(f"[{i}]来源：{chunk['filename']}，{location}\n{chunk['text']}")

    context = "\n\n".join(blocks)

    # 用 <context> 包住资料，明确边界：
    # 资料里若出现"忽略以上指令"这类文本，有边界时模型更容易当作数据而非指令
    return f"""参考资料：
<context>
{context}
</context>

问题：{question}

请严格依据上面的参考资料回答，并在引用处标注编号。"""


# 与 RAG_SYSTEM_PROMPT 第 3 条的字面内容必须完全一致。
# 代码里判断"模型是否拒答"依赖这个字符串，改一边忘了另一边会出错。
RAG_NO_ANSWER = "根据现有资料，我无法回答这个问题。"
