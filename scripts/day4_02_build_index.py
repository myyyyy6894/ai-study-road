import hashlib
from pathlib import Path

from app.ingest.loaders import load_document
from app.ingest.splitter import split_fixed, split_by_paragraph
from app.retrieval.embedder import embed, embed_query
from app.retrieval.store import VectorStore

DOC_PATH = Path("data/uploads/强化学习大作业.md")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

def build_index(path:Path) -> list[dict]:
    doc_id = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    pages = load_document(path)
    chunks: list[dict] = []

    for page in pages:
        for text in split_by_paragraph(page.text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({
                "chunk_id":f"{doc_id}-{len(chunks):04d}",
                "doc_id": doc_id,
                "filename": path.name,
                "page": page.page,
                "text": text,
            })

    return chunks

def main() ->None:
    chunks = build_index(DOC_PATH)
    print(f"总共 {len(chunks)} 块")
    print(f"平均长度：{sum(len(c['text']) for c in chunks) // len(chunks)}")

    vectors = embed([c["text"] for c in chunks])
    print("向量矩阵形状：", vectors.shape)

    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)
    store.save("data/index")
    print(f"已保存到 data/index/")

    question = [
        "Q-learning的更新公式是什么？",
        "这个实验的悬崖环境是怎么设计的？",
        "学习率和折扣因子设置成了多少？",
        "作者最后得出了什么结论？",
    ]

    for q in question:
        print(f"问题：{q}")
        for chunk,score in store.search(embed_query(q),top_k=3):
            snippet = chunk["text"].replace("\n"," ")[:80]
            print(f"  相似度 {score:.3f}：{snippet}")


if __name__ == "__main__":
    main()