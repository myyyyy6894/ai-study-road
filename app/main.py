"""FastAPI 入口。

Day 1 只挂一个最小接口，验证「HTTP → Pydantic → LLM → JSON」这条线。
后续每天往这里加路由，最终会拆成 app/api/*.py。
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.llm import chat
from app.schemas import ChatRequest, ChatResponse, Usage
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router


app = FastAPI(
    title="RAG 知识库问答 Demo",
    version="0.1.0",
    description="Day 1：LLM 调用与最小接口",
)



app.include_router(documents_router)
app.include_router(chat_router)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    # include_in_schema=False：给人看的页面，不必出现在 /docs 里
    return FileResponse("app/static/index.html")

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
