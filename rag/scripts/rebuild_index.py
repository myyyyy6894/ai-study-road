"""重建整个索引：扫描 data/uploads，去重 → 解析 → 切片 → 向量化 → 落盘。

为什么需要它：改了切片参数、清洗规则或 embedding 模型之后，
旧索引已经失效，必须全量重建。
"""

import hashlib

from app.config import settings
from app.ingest.loaders import load_document
from app.ingest.splitter import split_by_paragraph
from app.retrieval.embedder import embed
from app.retrieval.store import VectorStore

UPLOAD_DIR = settings.upload_dir
INDEX_DIR = settings.index_dir
ALLOWED = {".pdf", ".md", ".txt"}
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap


def main() -> None:
    seen: set[str] = set()
    all_chunks: list[dict] = []

    for path in sorted(UPLOAD_DIR.iterdir()):
        if path.suffix.lower() not in ALLOWED:
            continue

        # 用内容哈希去重：uploads 里可能有同一份文件的多个副本
        doc_id = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        if doc_id in seen:
            print(f"跳过重复内容：{path.name}")
            continue
        seen.add(doc_id)

        pages = load_document(path)
        before = len(all_chunks)

        for page in pages:
            for text in split_by_paragraph(page.text, CHUNK_SIZE, CHUNK_OVERLAP):
                all_chunks.append(
                    {
                        "chunk_id": f"{doc_id}-{len(all_chunks):04d}",
                        "doc_id": doc_id,
                        "filename": path.name,
                        "page": page.page,
                        "text": text,
                    }
                )

        print(f"{path.name}：{len(all_chunks) - before} 块")

    vectors = embed([c["text"] for c in all_chunks])
    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, all_chunks)
    store.save(INDEX_DIR)

    print(f"\n总计 {len(all_chunks)} 块，向量形状 {vectors.shape}，已写入 {INDEX_DIR}/")


if __name__ == "__main__":
    main()
