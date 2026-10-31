import asyncio

from app.config import settings
from app.llm import chat
from app.prompts import RAG_NO_ANSWER, RAG_SYSTEM_PROMPT, build_rag_prompt
from app.retrieval.embedder import embed_query
from app.retrieval.store import VectorStore

DEFAULT_TOP_K = settings.top_k
# 实测标定：库外问题最高 0.451，库内问题低至 0.464。
# 取 0.45 能同时保住召回和拒答（评估集上拒答率 100%）。
# 换 embedding 模型或语料后必须重新测，这个数字不可移植。
DEFAULT_MIN_SCORE = settings.min_score

_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(dim=settings.embed_dim)
        _store.load(settings.index_dir)
    return _store


async def answer(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = None,
) -> dict:
    store = get_store()
    query_vector = embed_query(question)
    results = store.search(query_vector, top_k=top_k)

    threshold = DEFAULT_MIN_SCORE if min_score is None else min_score
    results = [(chunk, score) for chunk, score in results if score >= threshold]

    if not results:
        return {"answer": RAG_NO_ANSWER, "citations": [], "retrieved": 0}

    chunks = [chunk for chunk, _ in results]

    resp = await chat(
        [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": build_rag_prompt(question, chunks)},
        ],
        temperature=0.1,
    )
    answer_text = resp.choices[0].message.content or " "

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

    return {
        "answer": answer_text,
        "citations": citations,
        "retrieved": len(results),
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    }


if __name__ == "__main__":
    result = asyncio.run(answer("Q-learning的更新公式是什么？"))
    print(result["answer"])
    print("\n---引用---")
    for c in result["citations"]:
        print(f"[{c['index']}] {c['filename']} {c['page']} 分数 {c['score']}")
