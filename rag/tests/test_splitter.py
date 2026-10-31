"""切片算法的单元测试。"""

import pytest

from app.ingest.splitter import split_by_paragraph, split_fixed


class TestSplitFixed:
    def test_covers_entire_text(self):
        text = "abcdefghij" * 10
        size, overlap = 20, 5
        step = size - overlap
        chunks = split_fixed(text, size, overlap)

        # 第一块从 0 开始
        assert chunks[0] == text[:size]
        # 最后一块必须覆盖到结尾
        assert chunks[-1] == text[(len(chunks) - 1) * step :]

        # 每个字符至少要落在某一块里（不丢内容）
        covered: set[int] = set()
        for idx, chunk in enumerate(chunks):
            start = idx * step
            covered.update(range(start, start + len(chunk)))
        assert covered == set(range(len(text)))

    def test_adjacent_chunks_overlap(self):
        text = "0123456789" * 10
        chunks = split_fixed(text, size=20, overlap=5)
        # 第 1 块 [0,20)，第 2 块 [15,35)：后 5 个字符应与前一块末尾相同
        assert chunks[0][-5:] == chunks[1][:5]

    @pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, 10), (10, 20)])
    def test_invalid_params_raise(self, size, overlap):
        with pytest.raises(ValueError):
            split_fixed("abc", size, overlap)

    def test_text_shorter_than_size(self):
        chunks = split_fixed("abc", size=100, overlap=10)
        assert chunks == ["abc"]


class TestSplitByParagraph:
    def test_does_not_break_paragraph(self):
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = split_by_paragraph(text, max_size=100, overlap=10)
        # 三段都不超长，应合成一块，且每段完整
        assert len(chunks) == 1
        assert "第一段内容。" in chunks[0]

    def test_respects_max_size(self):
        text = "\n\n".join(["每段大约十个字符左右的内容。" for _ in range(20)])
        chunks = split_by_paragraph(text, max_size=100, overlap=10)
        # 允许超出一点：单个段落本身可能就超过 max_size
        assert all(len(c) <= 130 for c in chunks)

    def test_long_paragraph_falls_back_to_fixed(self):
        long_para = "字" * 500
        chunks = split_by_paragraph(long_para, max_size=100, overlap=20)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_empty_text(self):
        assert split_by_paragraph("", max_size=100, overlap=10) == []
