import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.llm import chat_stream
from app.prompts import RAG_NO_ANSWER, RAG_SYSTEM_PROMPT, build_rag_prompt
from app.rag.memory import SessionStore, condense_question
from app.rag.pipeline import DEFAULT_MIN_SCORE, answer, get_store
from app.retrieval.embedder import embed_query
from app.schemas import RAGChatRequest, RAGChatResponse, Usage

router = APIRouter(prefix="/api", tags=["chat"])

sessions = SessionStore()


@router.post("/chat", response_model=RAGChatResponse, summary="知识库问答")
async def chat_rag(req: RAGChatRequest) -> RAGChatResponse:
    history = sessions.get(req.session_id)

    # ① 先改写（第一轮或关闭开关时直接返回原问题）
    rewritten = await condense_question(req.question, history, enabled=req.enable_rewrite)

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


def _sse(event: str, data: dict) -> str:
    """拼一条 SSE 消息。

    格式固定为 "event: 名字\ndata: json\n\n"：
    - 每条消息以空行结尾，前端才知道这一条结束了
    - data 必须是字符串，所以复杂对象要 json.dumps
    - ensure_ascii=False 让中文原样输出，否则前端收到 你 这种转义
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream", summary="流式问答（SSE）")
async def chat_stream_rag(req: RAGChatRequest):
    history = sessions.get(req.session_id)
    rewritten = await condense_question(req.question, history, enabled=req.enable_rewrite)
    rewritten_note = rewritten if rewritten != req.question else None

    store = get_store()
    results = [
        (chunk, score)
        for chunk, score in store.search(embed_query(rewritten), top_k=req.top_k)
        if score >= DEFAULT_MIN_SCORE
    ]

    async def event_generator():
        # 引用先发：检索已经完成，没必要等生成结束才让用户看到
        citations = [
            {
                "index": i,
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "page": chunk.get("page"),
                "text": chunk["text"],
                "score": round(score, 4),
            }
            for i, (chunk, score) in enumerate(results, start=1)
        ]
        yield _sse(
            "citations",
            {"citations": citations, "rewritten_question": rewritten_note},
        )

        # 检索为空：不调 LLM，直接给固定话术
        if not results:
            sessions.append(req.session_id, "user", req.question)
            sessions.append(req.session_id, "assistant", RAG_NO_ANSWER)
            yield _sse("delta", {"text": RAG_NO_ANSWER})
            yield _sse("done", {"answer": RAG_NO_ANSWER})
            return

        chunks = [chunk for chunk, _ in results]
        collected: list[str] = []

        async for delta in chat_stream(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": build_rag_prompt(rewritten, chunks)},
            ],
            temperature=0.1,
        ):
            collected.append(delta)
            yield _sse("delta", {"text": delta})

        # 流结束后才能拼出完整回答，这时才写历史
        answer_text = "".join(collected)
        sessions.append(req.session_id, "user", req.question)
        sessions.append(req.session_id, "assistant", answer_text)
        yield _sse("done", {"answer": answer_text})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        # 告诉中间的代理不要缓冲，否则"流式"会变成"等半天一次性出现"
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
