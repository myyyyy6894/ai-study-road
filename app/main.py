"""FastAPI 入口。

Day 1 只挂一个最小接口，验证「HTTP → Pydantic → LLM → JSON」这条线。
后续每天往这里加路由，最终会拆成 app/api/*.py。
"""

from fastapi import FastAPI

from app.llm import chat
from app.schemas import ChatRequest, ChatResponse, Usage

app = FastAPI(
    title="RAG 知识库问答 Demo",
    version="0.1.0",
    description="Day 1：LLM 调用与最小接口",
)

SYSTEM = "你是一个简洁的中文技术助手，回答控制在 3 句话以内。"


@app.post("/api/chat/simple", response_model=ChatResponse, summary="最小问答接口")
async def chat_simple(req: ChatRequest) -> ChatResponse:
    resp = await chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": req.question},
        ]
    )
    return ChatResponse(
        answer=resp.choices[0].message.content or "",
        usage=Usage(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
        ),
    )


@app.get("/health", summary="健康检查")
async def health() -> dict:
    return {"status": "ok"}
