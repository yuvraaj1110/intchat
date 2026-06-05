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


class _FakeDocWithProvenance:
    def __init__(self, text, topic, source_url, fetched_at):
        self.page_content = text
        self.metadata = {
            "topic": topic,
            "source_url": source_url,
            "fetched_at": fetched_at,
        }


def test_format_context_includes_provenance():
    docs = [
        _FakeDocWithProvenance(
            "OPT lasts 12 months.",
            "OPT",
            "https://www.uscis.gov/opt",
            "2026-06-05",
        ),
    ]
    ctx = chain.format_context(docs)
    assert "https://www.uscis.gov/opt" in ctx
    assert "2026-06-05" in ctx
    assert "OPT lasts 12 months." in ctx


def test_memory_window_keeps_last_n_pairs():
    mem = chain.ConversationWindow(max_pairs=2)
    mem.add("q1", "a1")
    mem.add("q2", "a2")
    mem.add("q3", "a3")
    rendered = mem.render()
    assert "q1" not in rendered   # evicted
    assert "q2" in rendered
    assert "q3" in rendered


class _Chunk:
    def __init__(self, content):
        self.content = content


class _FakeStore:
    """Minimal store whose retriever returns one doc for any query."""

    def similarity_search(self, query, k=5):
        return [_FakeDoc("OPT lets you work 12 months.", "OPT")]


class _FakeStreamingLLM:
    """Yields the answer in pieces, like ChatGroq.stream()."""

    def stream(self, messages):
        for piece in ["OPT ", "lets you ", "work."]:
            yield _Chunk(piece)


def test_answer_stream_yields_pieces_and_stores_memory():
    rag = chain.RAGChain(_FakeStore(), llm=_FakeStreamingLLM())
    pieces = list(rag.answer_stream("Can I work on OPT?"))
    assert pieces == ["OPT ", "lets you ", "work."]
    # Full answer accumulated into memory
    rendered = rag.memory.render()
    assert "Can I work on OPT?" in rendered
    assert "OPT lets you work." in rendered


def test_answer_collects_stream_into_string():
    rag = chain.RAGChain(_FakeStore(), llm=_FakeStreamingLLM())
    assert rag.answer("Can I work on OPT?") == "OPT lets you work."
