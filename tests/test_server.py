# tests/test_server.py
from unittest.mock import patch

from fastapi.testclient import TestClient


def _client():
    # Patch the heavy RAGChain/store setup so tests are fast and offline.
    with patch("app.server.load_store"), patch("app.server.RAGChain"):
        from app import server
        return TestClient(server.app)


def test_home_serves_html():
    client = _client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "International Student" in resp.text


def test_stream_empty_query_returns_400():
    client = _client()
    resp = client.get("/chat/stream?q=")
    assert resp.status_code == 400


def test_stream_whitespace_query_returns_400():
    client = _client()
    resp = client.get("/chat/stream?q=%20%20")
    assert resp.status_code == 400


def test_stream_yields_tokens_and_sources():
    class _FakeDoc:
        page_content = "OPT lets you work."
        metadata = {
            "topic": "OPT",
            "source_url": "https://uscis.gov/opt",
            "source_name": "USCIS",
            "fetched_at": "2026-06-06",
        }

    class _FakeRag:
        def retrieve(self, q):
            return [_FakeDoc()]

        def answer_stream_stateless(self, q, docs=None):
            yield "OPT "
            yield "answer."

    with patch("app.server.load_store"), patch("app.server.RAGChain"):
        from app import server
        server._rag = _FakeRag()
        client = TestClient(server.app)
        resp = client.get("/chat/stream?q=can+i+work")
        body = resp.text
        server._rag = None  # reset for other tests

    assert resp.status_code == 200
    assert "OPT " in body
    assert "answer." in body
    assert "uscis.gov/opt" in body   # sources event
    assert "USCIS" in body


def test_rate_limit_triggers():
    class _FakeDoc:
        page_content = "x"
        metadata = {}

    class _FakeRag:
        def retrieve(self, q):
            return [_FakeDoc()]

        def answer_stream_stateless(self, q, docs=None):
            yield "ok"

    from sse_starlette.sse import AppStatus
    with patch("app.server.load_store"), patch("app.server.RAGChain"):
        from app import server
        server._rag = _FakeRag()
        client = TestClient(server.app)
        # Hammer past the limit (20/hour); 25 calls should hit 429 at least once.
        statuses = []
        for _ in range(25):
            # TestClient spins a fresh event loop per call; reset sse-starlette's
            # module-global exit event so it rebinds to each loop (test-only quirk).
            AppStatus.should_exit_event = None
            statuses.append(client.get("/chat/stream?q=hi").status_code)
        server._rag = None
    assert 429 in statuses


def test_feedback_writes_log(tmp_path):
    log_file = tmp_path / "feedback.jsonl"
    with patch("app.server.load_store"), patch("app.server.RAGChain"):
        from app import server
        server.config.FEEDBACK_LOG = log_file
        client = TestClient(server.app)
        resp = client.post("/feedback", json={
            "question": "Can I work on OPT?",
            "answer": "Yes, 12 months.",
            "rating": "up",
        })
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    assert "OPT" in lines[0]
    assert "up" in lines[0]


def test_feedback_rejects_bad_rating():
    with patch("app.server.load_store"), patch("app.server.RAGChain"):
        from app import server
        client = TestClient(server.app)
        resp = client.post("/feedback", json={
            "question": "q", "answer": "a", "rating": "maybe",
        })
    assert resp.status_code == 400
