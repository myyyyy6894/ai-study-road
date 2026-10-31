"""文档管理接口：上传 / 列表 / 删除。"""

import asyncio
import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from app.config import settings
from app.ingest.loaders import load_document
from app.ingest.splitter import split_by_paragraph
from app.rag.pipeline import get_store
from app.retrieval.embedder import embed

router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_DIR = settings.upload_dir
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_BYTES = 20 * 1024 * 1024
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap


@router.post("", summary="上传文档并建立索引")
async def upload_document(file: UploadFile) -> dict:
    suffix = Path(file.filename or "").suffix.lower()

    # 白名单校验：不要相信客户端给的文件名和类型
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：{suffix}")

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="文件过大")

    doc_id = hashlib.md5(content).hexdigest()[:8]

    dest = UPLOAD_DIR / f"{doc_id}{suffix}"
    dest.write_bytes(content)

    pages = load_document(dest)
    chunks: list[dict] = []

    for page in pages:
        for text in split_by_paragraph(page.text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}-{len(chunks):04d}",
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "page": page.page,
                    "text": text,
                }
            )

    # 没有文本层（扫描件）时 chunks 为空，必须明确报错而不是静默成功
    if not chunks:
        raise HTTPException(status_code=400, detail="未提取到文本，可能是扫描件或空文档")

    # embed 是同步阻塞调用，丢到线程池，避免卡住事件循环
    vectors = await asyncio.to_thread(embed, [c["text"] for c in chunks])

    store = get_store()
    store.add(vectors, chunks)
    store.save("data/index")

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "num_chunks": len(chunks),
    }


@router.get("", summary="列出已入库文档")
async def list_documents() -> list[dict]:
    store = get_store()

    # chunks 是扁平列表，按 doc_id 聚合成"一个文档一条"
    docs: dict[str, dict] = {}

    for chunk in store.chunks:
        info = docs.setdefault(
            chunk["doc_id"],
            {"doc_id": chunk["doc_id"], "filename": chunk["filename"], "num_chunks": 0},
        )
        info["num_chunks"] += 1

    return list(docs.values())


@router.delete("/{doc_id}", summary="删除文档及其向量")
async def delete_document(doc_id: str) -> dict:
    store = get_store()

    # 先删向量，再删文件：如果删文件失败，至少索引是一致的
    removed = store.remove_doc(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_id}")

    store.save("data/index")

    for path in UPLOAD_DIR.glob(f"{doc_id}.*"):
        path.unlink(missing_ok=True)

    return {"doc_id": doc_id, "removed_chunks": removed}
