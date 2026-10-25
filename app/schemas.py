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
