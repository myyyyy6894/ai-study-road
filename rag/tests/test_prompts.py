"""Prompt 组装的单元测试。"""

from app.prompts import RAG_NO_ANSWER, build_rag_prompt

CHUNKS = [
    {"filename": "a.md", "page": None, "text": "第一段内容"},
    {"filename": "b.pdf", "page": 8, "text": "第二段内容"},
]


class TestBuildRagPrompt:
    def test_numbering_starts_at_one(self):
        prompt = build_rag_prompt("问题", CHUNKS)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[0]" not in prompt

    def test_none_page_shows_full_text(self):
        prompt = build_rag_prompt("问题", CHUNKS)
        # md/txt 没有页码，不能出现 "None"
        assert "None" not in prompt
        assert "全文" in prompt

    def test_page_number_rendered(self):
        prompt = build_rag_prompt("问题", CHUNKS)
        assert "第 8 页" in prompt

    def test_context_wrapped(self):
        prompt = build_rag_prompt("问题", CHUNKS)
        assert prompt.count("<context>") == 1
        assert prompt.count("</context>") == 1

    def test_question_included(self):
        prompt = build_rag_prompt("Q-learning 的公式是什么？", CHUNKS)
        assert "Q-learning 的公式是什么？" in prompt

    def test_chunks_separated_by_blank_line(self):
        prompt = build_rag_prompt("问题", CHUNKS)
        # 两条资料之间应有空行，模型对分段更敏感
        assert "第一段内容\n\n[2]" in prompt


def test_no_answer_constant_matches_prompt_rule():
    """拒答话术必须同时出现在 system prompt 和代码常量里。"""
    from app.prompts import RAG_SYSTEM_PROMPT

    assert RAG_NO_ANSWER in RAG_SYSTEM_PROMPT
