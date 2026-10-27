from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

@dataclass #装饰器
class Page:
    text:str
    page:int | None

def load_document(path:str | Path)->list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)

    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        return [Page(text=text, page=None)]

    raise ValueError(f"不支持的格式: {suffix}")

def _load_pdf(path:Path)->list[Page]:
    reader = PdfReader(str(path))
    pages:list[Page] = []

    for number, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""
        pages.append(Page(text=text,page=number))

    return pages
if __name__ == "__main__":
    upload_dir = Path("data/uploads")
    for path in sorted(upload_dir.iterdir()):
        if path.suffix.lower() not in {".pdf",".md",".txt"}:
            continue
        pages = load_document(path)
        lengths = [len(p.text) for p in pages]
        print(f"\n{path.name}")
        print(f"页数：{len(pages)}  总字符：{sum(lengths)}")
        print(
            f"每页字符：最小{min(lengths)} / 最大 {max(lengths)}"
            f"/平均 {sum(lengths) // len(lengths)}"
            )

        empty = [p.page for p in pages if len(p.text.strip()) < 10]
        if empty:
            print(f"内容过少的页: {empty}")


