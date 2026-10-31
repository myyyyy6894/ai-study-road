from app.llm import chat

# ============================================================
# 1. 问题改写提示词模板
# ============================================================

CONDENSE_TEMPLATE = """
根据对话历史，把用户最新的问题改写成一个独立、完整、不依赖上下文的问题。

要求：
- 保留原问题的意图
- 补全代词和省略的信息
- 只输出改写后的问题，不要任何解释、不要加引号

对话历史：
{history}

最新问题：{question}

改写后的问题
"""


# ============================================================
# 2. 会话历史存储
# ============================================================


class SessionStore:
    def __init__(self) -> None:
        # 使用字典保存不同会话的聊天记录
        # key：session_id（会话 ID）
        # value：该会话对应的消息列表
        self._session: dict[str, list[dict]] = {}

    def get(self, session_id: str) -> list[dict]:
        # 获取指定会话的聊天记录
        # 如果该 session_id 不存在，则自动创建一个空列表
        return self._session.setdefault(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        # 向指定会话追加一条消息
        # role：消息角色，例如 user 或 assistant
        # content：消息具体内容
        self._session.setdefault(session_id, []).append({"role": role, "content": content})

    def clear(self, session_id: str) -> None:
        # 删除指定会话的聊天记录
        # 如果会话不存在，也不会报错
        self._session.pop(session_id, None)


# ============================================================
# 3. 格式化历史对话
# ============================================================


def format_history(history: list[dict], max_turns: int = 3) -> str:

    # 只保留最近 max_turns 轮对话
    # 一轮通常包含一条 user 消息和一条 assistant 消息
    # 因此最多保留 max_turns * 2 条消息
    recent = history[-max_turns * 2 :]

    # 将消息列表转换成字符串
    # 例如：
    # user: 什么是 RAG？
    # assistant: RAG 是一种检索增强生成技术。
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in recent)


# ============================================================
# 4. 根据历史对话改写最新问题
# ============================================================


async def condense_question(
    question: str,
    history: list[dict],
    enabled: bool = True,
) -> str:

    # 如果没有历史记录，或者关闭了问题改写功能
    # 就直接返回用户原始问题，不需要调用大模型
    if not history or not enabled:
        return question

    # 调用大模型，让模型结合历史对话改写当前问题
    resp = await chat(
        [
            {
                "role": "user",
                "content": CONDENSE_TEMPLATE.format(
                    # 将最近的历史消息转换为文本
                    history=format_history(history),
                    # 将用户最新问题填入提示词
                    question=question,
                ),
            }
        ],
        temperature=0.0,  # 降低随机性，让改写结果更加稳定
        max_tokens=512,  # 限制模型最大输出 Token 数量
    )

    # 获取模型返回的文本，并去除首尾空白字符
    rewritten = (resp.choices[0].message.content or "").strip()

    # 去除模型可能添加的引号
    rewritten = rewritten.strip(" \\\"'“”")

    # 如果模型返回了有效内容，就返回改写后的问题
    # 如果模型返回空字符串，就退回到原始问题
    return rewritten or question
