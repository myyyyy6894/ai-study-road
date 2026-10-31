"""请求 / 响应模型。

为什么单独一层：
- 请求体自动校验（字段缺失、类型错误、超长 → 422，不用手写 if）
- /docs 里自动生成接口文档和调试表单
- 响应结构有契约，前端不用猜
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="用户问题")


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    answer: str
    usage: Usage | None = None


class RAGChatRequest(BaseModel):
    session_id: str = Field(default="default", description="会话 ID，相同 ID 共享历史")
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20, description="检索片段数")
    enable_rewrite: bool = Field(default=True, description="是否做多轮 query 改写")


class Citation(BaseModel):
    index: int  # 对应回答里的 [1]
    chunk_id: str
    filename: str
    page: int | None = None
    text: str
    score: float


class RAGChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation]
    retrieved: int
    # 只有发生改写时才返回，方便前端显示"我理解成了……"
    rewritten_question: str | None = None
    usage: Usage | None = None
