# tests/test_chain.py
from app import chain


class _FakeDoc:
    def __init__(self, text, topic):
        self.page_content = text
        self.metadata = {"topic": topic}


def test_format_context_lists_topics():
    docs = [
        _FakeDoc("OPT lets you work 12 months.", "Optional Practical Training (OPT)"),
        _FakeDoc("CPT is part of your curriculum.", "Curricular Practical Training (CPT)"),
    ]
    ctx = chain.format_context(docs)
    assert "OPT lets you work 12 months." in ctx
    assert "Optional Practical Training (OPT)" in ctx
    assert "CPT is part of your curriculum." in ctx


def test_memory_window_keeps_last_n_pairs():
    mem = chain.ConversationWindow(max_pairs=2)
    mem.add("q1", "a1")
    mem.add("q2", "a2")
    mem.add("q3", "a3")
    rendered = mem.render()
    assert "q1" not in rendered   # evicted
    assert "q2" in rendered
    assert "q3" in rendered
