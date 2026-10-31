from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass  # 装饰器
class Page:
    text: str
    page: int | None


def load_document(path: str | Path) -> list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)

    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        return [Page(text=text, page=None)]

    raise ValueError(f"不支持的格式: {suffix}")


def _load_pdf(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    pages: list[Page] = []

    for number, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""
        pages.append(Page(text=text, page=number))

    return _strip_repeated_lines(pages)


def _strip_repeated_lines(pages: list[Page], threshold: float = 0.5) -> list[Page]:
    """删除页眉、页脚这类"几乎每页都出现"的行。

    为什么必须做：幻灯片/论文的页眉会在每个 chunk 里重复出现，
    变成所有片段的公共前缀，把真正的内容"稀释"掉，导致相似度整体偏低。

    判定标准：某一行在超过 threshold 比例的页里出现，就认为是模板文字。
    用 set 对每页的行去重，避免同一页内重复行被重复计数。
    """
    if len(pages) < 3:
        return pages

    counter: dict[str, int] = {}
    for page in pages:
        for line in {s.strip() for s in page.text.splitlines() if s.strip()}:
            counter[line] = counter.get(line, 0) + 1

    repeated = {line for line, n in counter.items() if n / len(pages) > threshold}
    if not repeated:
        return pages

    cleaned: list[Page] = []
    for page in pages:
        lines = [s for s in page.text.splitlines() if s.strip() not in repeated]
        cleaned.append(Page(text="\n".join(lines).strip(), page=page.page))

    return cleaned


if __name__ == "__main__":
    upload_dir = Path("data/uploads")
    for path in sorted(upload_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".md", ".txt"}:
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
