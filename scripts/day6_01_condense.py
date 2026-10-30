import asyncio 
from app.rag.memory import condense_question
from app.rag.pipeline import get_store
from app.retrieval.embedder import embed_query

HISTORY = [
    {"role": "user", "content": "Q-learning 的更新公式是什么？"},
    {
        "role": "assistant",
        "content": "Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]",
    },
]

QUESTION = "那 α 是什么意思？"

async def main()-> None:
    rewritten = await condense_question(QUESTION, HISTORY)
    print(f"原始问题：{QUESTION}")
    print(f"改写后的问题：{rewritten}")

    store = get_store()
    for label,query in [("改写前",QUESTION),("改写后",rewritten)]:
        print(f"\n---{label}检索 Top - 3 ---")
        for chunk,score in store.search(embed_query(query),top_k=3):
            snippet = chunk["text"].replace("\n"," ")[:60]
            print(f"{score:.3f} {snippet}")

if __name__ == "__main__":
    asyncio.run(main())
