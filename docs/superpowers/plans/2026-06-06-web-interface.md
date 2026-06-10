# Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the existing RAG chatbot in a FastAPI web interface with a Q&A + source-sidebar UI, SSE token streaming, per-IP rate limiting, thumbs up/down feedback, and Railway deployment — all on free tiers.

**Architecture:** A thin FastAPI server (`app/server.py`) wraps the existing `RAGChain` as a transport layer. The RAG pipeline is unchanged except for two small additive methods (`retrieve`, `answer_stream_stateless`). The server is stateless per request (no cross-user memory). Static HTML/CSS/JS served directly — no build step. Deployed via Docker on Railway with Chroma baked into the image.

**Tech Stack:** FastAPI, uvicorn, sse-starlette, slowapi, vanilla HTML/CSS/JS, Docker, Railway.

**Branch:** `web-interface`

---

## File Structure

| File | Status | Responsibility |
|------|--------|----------------|
| `app/chain.py` | Modify | Add `retrieve()` and `answer_stream_stateless()` methods |
| `app/config.py` | Modify | Add `RATE_LIMIT`, `FEEDBACK_LOG` |
| `app/server.py` | Create | FastAPI app: routes, SSE, rate limiting, feedback |
| `app/static/index.html` | Create | Q&A page structure |
| `app/static/style.css` | Create | Two-column responsive layout |
| `app/static/app.js` | Create | SSE client, render answer + sidebar, feedback |
| `tests/test_server.py` | Create | Route, validation, rate-limit, feedback tests |
| `requirements.txt` | Modify | Add web deps |
| `.gitignore` | Modify | Add `feedback.jsonl` |
| `Dockerfile` | Create | Build image with Chroma baked in |
| `.dockerignore` | Create | Exclude venv, chroma_db, .env |
| `README.md` | Modify | Add web/deploy instructions |

---

## Task 1: Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add web dependencies to requirements.txt**

Append after the existing `pytest>=8.0.0` line:

```
fastapi>=0.110.0,<1
uvicorn[standard]>=0.29.0,<1
sse-starlette>=2.0.0,<3
slowapi>=0.1.9,<1
httpx>=0.27.0,<1
```

(`httpx` is needed by FastAPI's `TestClient` for the test suite.)

- [ ] **Step 2: Install**

Run: `pip install "fastapi>=0.110.0,<1" "uvicorn[standard]>=0.29.0,<1" "sse-starlette>=2.0.0,<3" "slowapi>=0.1.9,<1" "httpx>=0.27.0,<1"`
Expected: installs without error.

- [ ] **Step 3: Verify imports**

Run: `python3 -c "import fastapi, uvicorn, sse_starlette, slowapi, httpx; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add FastAPI web dependencies"
```

---

## Task 2: Config additions

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_web_config_present():
    from app import config
    assert config.RATE_LIMIT == "20/hour"
    assert str(config.FEEDBACK_LOG).endswith("feedback.jsonl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_web_config_present -v`
Expected: FAIL with `AttributeError: module 'app.config' has no attribute 'RATE_LIMIT'`

- [ ] **Step 3: Add config**

In `app/config.py`, after the `MAX_QUERY_LEN` line, add:

```python
# ── Web server ───────────────────────────────────────────────────────────
RATE_LIMIT = "20/hour"  # per-IP limit on /chat/stream
FEEDBACK_LOG = BASE_DIR / "feedback.jsonl"  # thumbs up/down log (gitignored)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_web_config_present -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add web server config (rate limit, feedback log)"
```

---

## Task 3: RAGChain stateless methods

**Files:**
- Modify: `app/chain.py`
- Test: `tests/test_chain.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_chain.py` (the `_FakeStore` and `_FakeStreamingLLM` classes already exist in that file):

```python
def test_retrieve_returns_docs():
    rag = chain.RAGChain(_FakeStore(), llm=_FakeStreamingLLM())
    docs = rag.retrieve("Can I work on OPT?")
    assert len(docs) >= 1
    assert docs[0].page_content


def test_answer_stream_stateless_does_not_touch_memory():
    rag = chain.RAGChain(_FakeStore(), llm=_FakeStreamingLLM())
    pieces = list(rag.answer_stream_stateless("Can I work on OPT?"))
    assert pieces == ["OPT ", "lets you ", "work."]
    # Memory must stay empty — stateless
    assert rag.memory.render() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chain.py::test_retrieve_returns_docs tests/test_chain.py::test_answer_stream_stateless_does_not_touch_memory -v`
Expected: FAIL with `AttributeError: 'RAGChain' object has no attribute 'retrieve'`

- [ ] **Step 3: Add the methods**

In `app/chain.py`, inside the `RAGChain` class, after the `answer` method, add:

```python
    def retrieve(self, question: str):
        """Return the retrieved documents for a question (for source display)."""
        return self.retriever.invoke(question)

    def answer_stream_stateless(self, question: str, docs=None):
        """Stream an answer without reading or writing conversation memory.

        Used by the web server, where many users share one RAGChain instance
        and must not share conversation history.
        """
        docs = docs if docs is not None else self.retrieve(question)
        context = format_context(docs)
        messages = prompts.build_prompt(
            context=context, chat_history="", question=question
        )
        for chunk in self.llm.stream(messages):
            piece = chunk.content if hasattr(chunk, "content") else str(chunk)
            if piece:
                yield piece
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chain.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/chain.py tests/test_chain.py
git commit -m "feat: add stateless retrieve + streaming methods to RAGChain"
```

---

## Task 4: FastAPI server — app skeleton & home route

**Files:**
- Create: `app/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_server.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_home_serves_html -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.server'`

- [ ] **Step 3: Create the server skeleton**

Create `app/server.py`:

```python
"""FastAPI web server wrapping the RAG chatbot for browser access.

Transport layer only — all model logic lives in RAGChain. The server is
stateless per request: each question is answered independently (no shared
conversation memory across concurrent users).
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.chat import load_store
from app.chain import RAGChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="International Student Assistant")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Single shared RAG instance (stateless usage only — see answer_stream_stateless).
_rag: RAGChain | None = None


def get_rag() -> RAGChain:
    global _rag
    if _rag is None:
        _rag = RAGChain(load_store())
    return _rag


@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))
```

- [ ] **Step 4: Create a minimal index.html so the route can serve it**

Create `app/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>International Student Assistant</title>
</head>
<body>
  <h1>International Student Assistant</h1>
</body>
</html>
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_server.py::test_home_serves_html -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/server.py app/static/index.html tests/test_server.py
git commit -m "feat: add FastAPI server skeleton + home route"
```

---

## Task 5: Query validation on the stream route

**Files:**
- Modify: `app/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
def test_stream_empty_query_returns_400():
    client = _client()
    resp = client.get("/chat/stream?q=")
    assert resp.status_code == 400


def test_stream_whitespace_query_returns_400():
    client = _client()
    resp = client.get("/chat/stream?q=%20%20")
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_server.py::test_stream_empty_query_returns_400 -v`
Expected: FAIL (404, route doesn't exist yet)

- [ ] **Step 3: Add the validated stream route**

In `app/server.py`, add the import at the top (with the other `from app` imports):

```python
from app.chat import load_store, validate_query
```

(Replace the existing `from app.chat import load_store` line with the line above.)

Then add this route at the end of the file:

```python
from fastapi import HTTPException


@app.get("/chat/stream")
def chat_stream(request: Request, q: str = ""):
    question = validate_query(q)
    if question is None:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    # Streaming response added in the next task.
    return {"ok": True, "question": question}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_server.py -k stream -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/server.py tests/test_server.py
git commit -m "feat: add query validation to /chat/stream"
```

---

## Task 6: SSE streaming of answer tokens + sources

**Files:**
- Modify: `app/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
def test_stream_yields_tokens_and_sources():
    # Build a fake RAG whose stateless stream yields known tokens and docs.
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

    assert resp.status_code == 200
    assert "OPT " in body
    assert "answer." in body
    assert "uscis.gov/opt" in body   # sources event
    assert "USCIS" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_stream_yields_tokens_and_sources -v`
Expected: FAIL (route currently returns JSON, not a stream with these contents)

- [ ] **Step 3: Implement the SSE stream**

In `app/server.py`, add the import near the top:

```python
import json
from sse_starlette.sse import EventSourceResponse
```

Replace the entire `chat_stream` function body with:

```python
@app.get("/chat/stream")
def chat_stream(request: Request, q: str = ""):
    question = validate_query(q)
    if question is None:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    rag = get_rag()

    def event_generator():
        try:
            docs = rag.retrieve(question)
            for piece in rag.answer_stream_stateless(question, docs=docs):
                yield {"event": "token", "data": piece}
            # Send unique sources once, after the answer.
            seen = set()
            sources = []
            for d in docs:
                url = d.metadata.get("source_url", "")
                if url and url not in seen:
                    seen.add(url)
                    sources.append({
                        "source_name": d.metadata.get("source_name", ""),
                        "source_url": url,
                        "fetched_at": d.metadata.get("fetched_at", ""),
                        "topic": d.metadata.get("topic", ""),
                    })
            yield {"event": "sources", "data": json.dumps(sources)}
            yield {"event": "done", "data": ""}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error answering question: %s", exc)
            yield {
                "event": "error",
                "data": "I'm having trouble right now, please try again in a moment.",
            }

    return EventSourceResponse(event_generator())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py::test_stream_yields_tokens_and_sources -v`
Expected: PASS

- [ ] **Step 5: Run full server tests**

Run: `pytest tests/test_server.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add app/server.py tests/test_server.py
git commit -m "feat: stream answer tokens + sources via SSE"
```

---

## Task 7: Rate limiting

**Files:**
- Modify: `app/server.py`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
def test_rate_limit_triggers():
    class _FakeDoc:
        page_content = "x"
        metadata = {}

    class _FakeRag:
        def retrieve(self, q):
            return [_FakeDoc()]

        def answer_stream_stateless(self, q, docs=None):
            yield "ok"

    with patch("app.server.load_store"), patch("app.server.RAGChain"):
        from app import server
        # Tighten the limit for the test.
        server.limiter._default_limits = []
        from importlib import reload
        server._rag = _FakeRag()
        client = TestClient(server.app)
        # Hammer past the limit (20/hour); 25 calls should hit 429 at least once.
        statuses = [client.get("/chat/stream?q=hi").status_code for _ in range(25)]
    assert 429 in statuses
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_server.py::test_rate_limit_triggers -v`
Expected: FAIL (no 429 — limiter not wired yet)

- [ ] **Step 3: Wire up slowapi**

In `app/server.py`, add imports near the top:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
```

After `app = FastAPI(...)` and before the static mount, add:

```python
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "You've asked a lot of questions! Please wait a bit before asking more."},
    )
```

Decorate the stream route — change its signature line to:

```python
@app.get("/chat/stream")
@limiter.limit(config.RATE_LIMIT)
def chat_stream(request: Request, q: str = ""):
```

(`slowapi` requires the `request: Request` parameter, which is already present.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_server.py::test_rate_limit_triggers -v`
Expected: PASS

- [ ] **Step 5: Run all server tests**

Run: `pytest tests/test_server.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add app/server.py tests/test_server.py
git commit -m "feat: add per-IP rate limiting to /chat/stream"
```

---

## Task 8: Feedback endpoint

**Files:**
- Modify: `app/server.py`
- Modify: `.gitignore`
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_server.py -k feedback -v`
Expected: FAIL (404, route doesn't exist)

- [ ] **Step 3: Add the feedback route**

In `app/server.py`, add near the other imports:

```python
from datetime import datetime, timezone
from pydantic import BaseModel
```

Add this model and route at the end of the file:

```python
class Feedback(BaseModel):
    question: str
    answer: str
    rating: str


@app.post("/feedback")
def feedback(item: Feedback):
    if item.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": item.question[:1000],
        "answer": item.answer[:5000],
        "rating": item.rating,
    }
    with open(config.FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_server.py -k feedback -v`
Expected: PASS

- [ ] **Step 5: Gitignore the feedback log**

Add to `.gitignore`:

```
feedback.jsonl
```

- [ ] **Step 6: Commit**

```bash
git add app/server.py tests/test_server.py .gitignore
git commit -m "feat: add thumbs up/down feedback endpoint"
```

---

## Task 9: Frontend — full UI (HTML + CSS + JS)

**Files:**
- Modify: `app/static/index.html`
- Create: `app/static/style.css`
- Create: `app/static/app.js`

- [ ] **Step 1: Write the full index.html**

Replace `app/static/index.html` entirely:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>International Student Assistant</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>🎓 International Student Assistant</h1>
    <p class="subtitle">Ask about F-1, OPT, CPT, SEVIS, employment, and student life in the U.S.</p>
  </header>

  <main>
    <section class="chat">
      <form id="ask-form">
        <input id="question" type="text" placeholder="Type your question..." autocomplete="off" />
        <button type="submit" id="send-btn">Ask</button>
      </form>

      <div id="answer" class="answer"></div>

      <div id="feedback" class="feedback" hidden>
        <span>Was this helpful?</span>
        <button data-rating="up" class="thumb">👍</button>
        <button data-rating="down" class="thumb">👎</button>
        <span id="feedback-thanks" hidden>Thanks for the feedback!</span>
      </div>
    </section>

    <aside class="sidebar">
      <h2>Sources</h2>
      <div id="sources" class="sources">
        <p class="sources-empty">Sources will appear here after you ask a question.</p>
      </div>
    </aside>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write style.css**

Create `app/static/style.css`:

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: #1a1a2e; background: #f4f6fb; line-height: 1.6;
}
header {
  background: #16213e; color: #fff; padding: 1.5rem 2rem;
}
header h1 { font-size: 1.4rem; }
.subtitle { font-size: 0.9rem; opacity: 0.8; margin-top: 0.25rem; }

main {
  display: grid; grid-template-columns: 1fr 320px; gap: 1.5rem;
  max-width: 1100px; margin: 1.5rem auto; padding: 0 1.5rem;
}

.chat { background: #fff; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
#ask-form { display: flex; gap: 0.5rem; margin-bottom: 1.25rem; }
#question {
  flex: 1; padding: 0.75rem 1rem; border: 1px solid #d0d5dd;
  border-radius: 8px; font-size: 1rem;
}
#question:focus { outline: none; border-color: #16213e; }
#send-btn {
  padding: 0.75rem 1.5rem; background: #16213e; color: #fff;
  border: none; border-radius: 8px; font-size: 1rem; cursor: pointer;
}
#send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.answer { white-space: pre-wrap; min-height: 2rem; font-size: 1rem; }
.answer.error { color: #b42318; }

.feedback { margin-top: 1.5rem; display: flex; align-items: center; gap: 0.5rem; font-size: 0.9rem; color: #555; }
.thumb { background: none; border: 1px solid #d0d5dd; border-radius: 6px; padding: 0.25rem 0.6rem; cursor: pointer; font-size: 1rem; }
.thumb:hover { background: #f0f2f7; }

.sidebar { background: #fff; border-radius: 12px; padding: 1.25rem; box-shadow: 0 1px 4px rgba(0,0,0,0.06); height: fit-content; }
.sidebar h2 { font-size: 1rem; margin-bottom: 0.75rem; color: #16213e; }
.source-item { border-left: 3px solid #16213e; padding: 0.5rem 0.75rem; margin-bottom: 0.75rem; background: #f8f9fc; border-radius: 4px; }
.source-item .name { font-weight: 600; font-size: 0.88rem; }
.source-item a { color: #2d5bd7; font-size: 0.8rem; word-break: break-all; text-decoration: none; }
.source-item a:hover { text-decoration: underline; }
.source-item .date { color: #888; font-size: 0.75rem; }
.sources-empty { color: #999; font-size: 0.85rem; }

@media (max-width: 768px) {
  main { grid-template-columns: 1fr; }
}
```

- [ ] **Step 3: Write app.js**

Create `app/static/app.js`:

```javascript
const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const feedbackEl = document.getElementById("feedback");
const thanksEl = document.getElementById("feedback-thanks");

let lastQuestion = "";
let lastAnswer = "";

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  askQuestion(question);
});

function askQuestion(question) {
  lastQuestion = question;
  lastAnswer = "";
  answerEl.textContent = "";
  answerEl.classList.remove("error");
  sourcesEl.innerHTML = "";
  feedbackEl.hidden = true;
  thanksEl.hidden = true;
  sendBtn.disabled = true;

  const es = new EventSource("/chat/stream?q=" + encodeURIComponent(question));

  es.addEventListener("token", (ev) => {
    lastAnswer += ev.data;
    answerEl.textContent = lastAnswer;
  });

  es.addEventListener("sources", (ev) => {
    const sources = JSON.parse(ev.data);
    renderSources(sources);
  });

  es.addEventListener("done", () => {
    es.close();
    sendBtn.disabled = false;
    if (lastAnswer.trim()) feedbackEl.hidden = false;
  });

  es.addEventListener("error", (ev) => {
    es.close();
    sendBtn.disabled = false;
    // ev.data is set for our explicit error event; network errors have none.
    answerEl.textContent = ev.data || "Something went wrong. Please try again.";
    answerEl.classList.add("error");
  });
}

function renderSources(sources) {
  if (!sources.length) {
    sourcesEl.innerHTML = '<p class="sources-empty">No external sources for this answer.</p>';
    return;
  }
  sourcesEl.innerHTML = sources.map((s) => `
    <div class="source-item">
      <div class="name">${escapeHtml(s.source_name || s.topic || "Source")}</div>
      <a href="${escapeHtml(s.source_url)}" target="_blank" rel="noopener">${escapeHtml(s.source_url)}</a>
      <div class="date">retrieved ${escapeHtml(s.fetched_at || "")}</div>
    </div>
  `).join("");
}

feedbackEl.addEventListener("click", (e) => {
  const btn = e.target.closest(".thumb");
  if (!btn) return;
  fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: lastQuestion, answer: lastAnswer, rating: btn.dataset.rating }),
  });
  thanksEl.hidden = false;
});

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
```

- [ ] **Step 4: Manual smoke test the UI**

Run (in a separate terminal, requires the Chroma store to exist and GROQ_API_KEY set):
`uvicorn app.server:app --reload --port 8000`
Open `http://localhost:8000`, ask "How many hours can I work on OPT?", and confirm:
- Answer streams in token by token
- Sources appear in the right sidebar with clickable URLs
- 👍/👎 buttons appear after the answer and show "Thanks" when clicked

- [ ] **Step 5: Commit**

```bash
git add app/static/
git commit -m "feat: add Q&A web UI with source sidebar and feedback"
```

---

## Task 10: Dockerfile & deployment config

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Create .dockerignore**

Create `.dockerignore`:

```
__pycache__/
*.pyc
.env
.git/
chroma_db/
feedback.jsonl
datasets/normalized_all.json
docs/
tests/
.pytest_cache/
```

- [ ] **Step 2: Create the Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System deps for lxml (used by trafilatura)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY datasets/ ./datasets/
COPY sources.yaml .

# Build the Chroma store at image-build time so it's baked into the image.
# (No GROQ_API_KEY needed for ingestion — only embeddings run here.)
RUN python -m app.build_kb --reset

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

- [ ] **Step 3: Build the image locally to verify**

Run: `docker build -t intchat .`
Expected: build completes; the `build_kb` step prints the chunk count near the end.

(If Docker is not installed locally, skip this step — Railway will build it. Note this in the commit message.)

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "build: add Dockerfile with Chroma baked in for Railway deploy"
```

---

## Task 11: Full verification, README, merge

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -q -m "not slow"`
Expected: all tests pass (44 existing + new server/chain/config tests).

- [ ] **Step 2: Update README with web + deploy instructions**

In `README.md`, after the Quickstart section, add a new section:

```markdown
## Running the web interface

Local:
```bash
uvicorn app.server:app --reload --port 8000
# open http://localhost:8000
```

Deploy to Railway (free tier):
1. Push to GitHub `main`.
2. On railway.app, create a new project from your repo.
3. Set the `GROQ_API_KEY` environment variable in Railway.
4. Railway builds the Dockerfile (Chroma is baked in) and gives you a public URL.

The web UI is **stateless** (no cross-question memory) and rate-limited to
20 questions per IP per hour to protect the shared Groq quota. Thumbs up/down
feedback is logged to `feedback.jsonl` on the server.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add web interface and Railway deploy instructions"
```

- [ ] **Step 4: Merge to main and push**

```bash
git checkout main && git merge web-interface && git push
```

- [ ] **Step 5: Final report**

Confirm: tests green, server runs locally, Docker builds (or noted as Railway-only), merged to main.

---

## Self-Review Notes

- **Spec coverage:** home route (T4), validation (T5), SSE tokens+sources (T6), rate limit (T7), feedback (T8), UI Q&A+sidebar+feedback (T9), Dockerfile/Railway (T10), stateless methods (T3), config (T2), deps (T1), README (T11). All spec sections covered.
- **Type consistency:** `retrieve()` and `answer_stream_stateless(question, docs=None)` defined in T3, used identically in T6. `Feedback` model fields (`question`, `answer`, `rating`) match the JS `fetch` body in T9. SSE event names (`token`, `sources`, `done`, `error`) match between T6 server and T9 client.
- **Stateless guarantee:** server uses only `retrieve` + `answer_stream_stateless`; never `answer_stream` (which mutates memory).
