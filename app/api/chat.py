from fastapi import APIRouter

from app.rag.memory import SessionStore, condense_question
from app.rag.pipeline import answer
from app.schemas import RAGChatRequest, RAGChatResponse, Usage


router = APIRouter(prefix="/api", tags=["chat"])

sessions = SessionStore()

@router.post("/chat", response_model=RAGChatResponse, summary="知识库问答")
async def chat_rag(req: RAGChatRequest) -> RAGChatResponse:
    history = sessions.get(req.session_id)

    # ① 先改写（第一轮或关闭开关时直接返回原问题）
    rewritten = await condense_question(
        req.question, history, enabled=req.enable_rewrite
    )

    # ② 用改写后的问题检索并生成
    result = await answer(rewritten, top_k=req.top_k)

    # ③ 历史里存"用户的原话"，不存改写结果：
    # 用户看到的对话记录必须和他输入的一致，改写只是内部手段
    sessions.append(req.session_id, "user", req.question)
    sessions.append(req.session_id, "assistant", result["answer"])

    # pipeline 只在真正调用了 LLM 时才返回 usage（拒答时没有）
    usage = None
    if "usage" in result:
        u = result["usage"]
        usage = Usage(
            prompt_tokens=u["prompt_tokens"],
            completion_tokens=u["completion_tokens"],
            total_tokens=u["prompt_tokens"] + u["completion_tokens"],
        )

    return RAGChatResponse(
        session_id=req.session_id,
        answer=result["answer"],
        citations=result["citations"],
        retrieved=result["retrieved"],
        # 没改写时不返回，避免前端显示一句和原文一样的问题
        rewritten_question=rewritten if rewritten != req.question else None,
        usage=usage,
    )

@router.get("/sessions/{session_id}", summary="查看会话历史")
async def get_session(session_id: str) -> dict:
    return {"session_id": session_id, "messages": sessions.get(session_id)}

@router.delete("/sessions/{session_id}", summary="清空会话")
async def clear_session(session_id: str) -> dict:
    sessions.clear(session_id)
    return {"ok": True}
