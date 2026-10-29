import asyncio

from app.llm import chat
from app.prompts import RAG_NO_ANSWER,build_rag_prompt,RAG_SYSTEM_PROMPT
from app.retrieval.embedder import embed_query
from app.retrieval.store import VectorStore

DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.5   # 实测：库内问题 0.59+，库外 0.45-；换模型/语料需重测

_store:VectorStore | None = None

def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load("data/index")
    return _store

async def answer(
        question:str,
        top_k : int = DEFAULT_TOP_K,
        min_score : float | None = None,
)->dict:
    store = get_store()
    query_vector = embed_query(question)
    results = store.search(query_vector,top_k=top_k)

    threshold = DEFAULT_MIN_SCORE if min_score is None else min_score
    results = [(chunk,score) for chunk,score in results if score >= threshold]

    if not results:
        return {"answer":RAG_NO_ANSWER,"citations":[] ,"retrieved":0}

    chunks = [chunk for chunk,_ in results]

    resp = await chat(
        [
            {"role":"system","content":RAG_SYSTEM_PROMPT},
            {"role":"user","content":build_rag_prompt(question,chunks)},
        ],
        temperature=0.1,
    )
    answer_text = resp.choices[0].message.content or " "

    citations = [
        {
            "index":i,
            "chunk_id":chunk["chunk_id"],
            "filename":chunk["filename"],
            "page":chunk.get("page"),
            "text":chunk["text"],
            "score":round(score,4)
        }
        for i,(chunk,score) in enumerate(results,start=1)
    ]

    return{
        "answer":answer_text,
        "citations":citations,
        "retrieved":len(results),
        "usage":{
            "prompt_tokens":resp.usage.prompt_tokens,
            "completion_tokens":resp.usage.completion_tokens,
        },
    }

if __name__ == "__main__":
    result = asyncio.run(answer("Q-learning的更新公式是什么？"))
    print(result["answer"])
    print("\n---引用---")
    for c in result["citations"]:
        print(f"[{c['index']}] {c['filename']} {c['page']} 分数 {c['score']}")
    
    

