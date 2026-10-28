#帮我把缩进处理一下
def split_fixed(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("size 必须大于 0")
    if overlap >= size:
        raise ValueError("overlap 必须小于 size")

    step = size - overlap

    chunks: list[str] = []
    start = 0  # 当前窗口起点

    while start < len(text):
        end = start + size

        # Python 切片越界不会报错：text[0:9999] 会自动截到末尾
        chunks.append(text[start:end])

        # 已经切到末尾就退出。
        if end >= len(text):
            break

        start += step

    return chunks


def split_by_paragraph(text: str, max_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(para) > max_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(split_fixed(para, max_size, overlap))
        elif not buffer:
            buffer = para
        elif len(buffer) + 2 + len(para) <= max_size:
            # +2 是拼接时插入的两个换行符（保持段落之间的空行），
            # 判断容量时也要算进去，否则每块会超出 max_size
            buffer = buffer + "\n\n" + para
        else:
            chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)
    return chunks

if __name__ == "__main__":
    from pathlib import Path

    text = Path("data/uploads/强化学习大作业.md").read_text(encoding="utf-8")

    for size, overlap in [(200, 40), (500, 100), (1000, 200)]:
        fixed = split_fixed(text, size, overlap)
        para = split_by_paragraph(text, size, overlap)
        print(f"\nsize={size} overlap={overlap}")
        print(f"  固定窗口：{len(fixed)} 块")
        print(f"  段落优先：{len(para)} 块")

    print("\n段落优先分块示例：")
    print(split_by_paragraph(text, 500, 100)[0][:200])
