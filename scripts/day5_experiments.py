import asyncio 
from app.llm import chat
from app.prompts import RAG_SYSTEM_PROMPT,build_rag_prompt
from app.rag.pipeline import answer,get_store
from app.retrieval.embedder import embed_query

Q_IN = "Q-learning 的更新公式是什么？"
Q_OUT = "量子计算机的 Shor 算法是如何分解大整数因子的？"
# 半相关：主题沾边，但文档里未必有具体答案——幻觉最容易出现在这里
Q_HALF = "这份实验是在什么型号的 CPU 和 GPU 上训练的？"

LAX_SYSTEM = "你是一个知识库问答助手，请依据参考资料回答问题。"

async def plain_llm(question:str)->str:
    resp = await chat(
        [
            {"role":"system","content":"你是一个知识渊博的助手"},
            {"role":"user","content":question},
        ],
        temperature=0.1,
    )

    return resp.choices[0].message.content

async def rag_with_system(question:str,system:str)->str:
    store = get_store()
    results = store.search(embed_query(question),top_k=4)
    prompt = build_rag_prompt(question,[chunk for chunk,_ in results])
    resp = await chat(
        [
            {"role":"system","content":system},
            {"role":"user","content":build_rag_prompt(question,[chunk for chunk,_ in results])},
        ],
        temperature=0.1,
    )

    return resp.choices[0].message.content or " "
async def main()->None:
    print("======实验A：库内问题======")
    print("\n[A1] 无检索，纯LLM")

    print(await plain_llm(Q_IN))
    print("\n[A2] 有检索，RAG")
    result = await answer(Q_IN)
    print(result["answer"])

    print(f" 检索分数：{[c['score'] for c in result['citations']]}")

    print("======实验B：库外问题======")
    print("\n[B1] 无检索，纯LLM")
    print(await plain_llm(Q_OUT))
    print("\n[B2] 有检索，RAG")
    result = await answer(Q_OUT)
    print(result["answer"])
    print(f" 检索分数：{[c['score'] for c in result['citations']]}")

    print("\n[B3] 有检索，去掉拒绝回答的系统提示")
    print(await rag_with_system(Q_OUT, LAX_SYSTEM))

    # B4：资料"看起来相关"但可能没有答案——最容易产生幻觉的区间
    print("\n[B4] 半相关问题，去掉拒答指令：")
    print(await rag_with_system(Q_HALF, LAX_SYSTEM))

    print("\n[B5] 半相关问题，严格拒答指令：")
    result = await answer(Q_HALF)
    print(result["answer"])
    print(f"  检索分数: {[c['score'] for c in result['citations']]}")


if __name__ == "__main__":
    asyncio.run(main())