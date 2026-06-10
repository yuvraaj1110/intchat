# Web Interface for Public Beta

**Date:** 2026-06-06
**Status:** Approved
**Goal:** Wrap the existing RAG chatbot in a web interface so non-technical international students (20-100 beta testers, strangers) can use it from a public URL. Add rate limiting, source-sidebar UI, thumbs up/down feedback, and Railway deployment — all on free tiers.

---

## Design Principles

1. **The RAG brain does not change.** The web server is a transport layer wrapped around the existing `RAGChain`. Model logic and HTTP logic stay fully separated.
2. **Stateless per request.** For the beta, each question is answered independently (no cross-question memory). Per-session memory is deferred to avoid concurrency complexity.
3. **Fail gracefully in the UI, loudly in the logs.** Users never see stack traces; technical detail goes to server logs.
4. **Free tier only.** Railway free credit + Groq free tier + local Chroma + local embeddings = $0.

---

## Architecture

```
Browser (student)
   │  HTTP
   ▼
FastAPI server (app/server.py)        ◄── NEW
   │  ├─ GET  /              → serves index.html
   │  ├─ GET  /chat/stream   → streams answer tokens (SSE) + sources
   │  └─ POST /feedback      → logs thumbs up/down
   ▼
RAGChain  (app/chain.py)              ◄── minimal addition (source capture)
   ▼
retriever → prompt → Groq LLM         ◄── unchanged
```

---

## New Files

| File | Responsibility |
|------|----------------|
| `app/server.py` | FastAPI app: routes, SSE streaming, rate limiting, feedback logging |
| `app/static/index.html` | Q&A page structure (question box, answer pane, sources sidebar) |
| `app/static/style.css` | Two-column layout, mobile-responsive (sidebar collapses below) |
| `app/static/app.js` | Send question via SSE, render streamed answer, populate sidebar, feedback buttons |
| `Dockerfile` | Build image, run `build_kb` at build time so Chroma is baked in |
| `.dockerignore` | Exclude venv, chroma_db, .env from image context |

## Changed Files

| File | Change |
|------|--------|
| `requirements.txt` | Add `fastapi`, `uvicorn[standard]`, `slowapi`, `sse-starlette` |
| `app/config.py` | Add `RATE_LIMIT`, `FEEDBACK_LOG` path |
| `app/chain.py` | Add `retrieve(question)` returning docs, so server can send sources to sidebar; keep `answer_stream` unchanged |

---

## API Endpoints

### `GET /`
Serves `app/static/index.html`. Static files mounted at `/static`.

### `GET /chat/stream?q=<question>`
Server-Sent Events stream.

Flow:
1. Rate limiter checks IP (20/hour). Over limit → HTTP 429 with friendly JSON.
2. Validate query (reuse `chat.validate_query`). Empty → 400.
3. Retrieve docs once: `docs = rag.retrieve(q)`.
4. Build context + stream tokens via the existing prompt/LLM path.
5. Emit events:
   - `event: token` for each text chunk
   - `event: sources` (once, at end) with JSON list of `{source_name, source_url, fetched_at, topic}`
   - `event: done` to close
6. On LLM error: emit `event: error` with a friendly message; log the real exception.

**Note on memory:** the server uses a fresh, stateless retrieval+generation per request. It does NOT use `ConversationWindow`. Each `/chat/stream` call is independent.

### `POST /feedback`
Body: `{question: str, answer: str, rating: "up" | "down"}`
Appends one JSON line to `FEEDBACK_LOG` (JSONL). Returns `{ok: true}`.
Validates rating is "up"/"down"; truncates question/answer to reasonable lengths.

---

## RAGChain Change

Add a method that exposes the retrieved docs so the server can render the sidebar:

```python
def retrieve(self, question: str) -> list:
    """Return the retrieved documents for a question (for source display)."""
    return self.retriever.invoke(question)
```

Add a stateless streaming method (no memory mutation) so concurrent users don't share history:

```python
def answer_stream_stateless(self, question: str, docs=None):
    """Stream an answer without reading or writing conversation memory."""
    docs = docs if docs is not None else self.retrieve(question)
    context = format_context(docs)
    messages = prompts.build_prompt(context=context, chat_history="", question=question)
    for chunk in self.llm.stream(messages):
        piece = chunk.content if hasattr(chunk, "content") else str(chunk)
        if piece:
            yield piece
```

The existing `answer_stream` (with memory) stays for the CLI.

---

## UI Layout

```
┌────────────────────────────────────────────────────────┐
│  🎓 International Student Assistant                       │
│  Ask about F-1, OPT, CPT, SEVIS, employment, and more   │
├──────────────────────────────────┬─────────────────────┤
│  [ Your question...        ][→]   │  SOURCES            │
│                                   │  📄 USCIS — OPT     │
│  <streamed answer text>           │     uscis.gov/...   │
│                                   │     retrieved 6/6   │
│  <legal disclaimer>               │  📄 Purdue ISS      │
│                                   │     purdue.edu/...  │
│  Was this helpful?  👍  👎        │     retrieved 6/6   │
└──────────────────────────────────┴─────────────────────┘
```

- Main pane: question input (top), streamed answer, always-visible disclaimer, feedback buttons.
- Sidebar: clickable source URLs + fetch dates from the `sources` SSE event.
- Mobile (<768px): sidebar stacks below the answer.
- No build step — plain HTML/CSS/JS served as static files.

---

## Rate Limiting

- Library: `slowapi` (in-memory, no Redis).
- Limit: `RATE_LIMIT = "20/hour"` per client IP, applied to `/chat/stream`.
- Over limit: friendly message ("You've asked a lot of questions! Please wait a bit before asking more."), HTTP 429.
- Rationale: protects the shared Groq free-tier quota from a single heavy user.

---

## Feedback Logging

- `POST /feedback` appends JSONL to `FEEDBACK_LOG` (default `feedback.jsonl`, gitignored).
- Each line: `{"ts": iso, "question": ..., "answer": ..., "rating": "up"|"down"}`.
- No database. Read the file to find low-quality answers.

---

## Error Handling

| Failure | User sees | Server does |
|---------|-----------|-------------|
| Groq down / timeout | "I'm having trouble right now, please try again in a moment." | Logs exception |
| Rate limit hit | "You've asked a lot of questions! Please wait a bit." | Returns 429 |
| Empty / invalid query | Gentle inline nudge, no request sent | Validation in JS + server 400 |
| Unexpected server error | Generic friendly message | Logs full traceback |

Users never see stack traces. All technical detail is logged server-side.

---

## Deployment (Railway)

- `Dockerfile`:
  - Base `python:3.12-slim`
  - Install requirements
  - Copy app + datasets + sources.yaml
  - Run `python -m app.build_kb --reset` at build time → Chroma baked into image
  - `CMD uvicorn app.server:app --host 0.0.0.0 --port $PORT`
- Railway watches GitHub `main` → auto-deploy on push.
- Secret: `GROQ_API_KEY` set in Railway env vars (never committed).
- Cost: $0 (Railway free credit + Groq free tier).

---

## Testing

`tests/test_server.py` (using FastAPI `TestClient` + a fake LLM, mirroring existing chain tests):
- `GET /` returns 200 and serves HTML
- `GET /chat/stream` with valid query streams token events + a sources event
- `GET /chat/stream` with empty query returns 400
- Rate limiter returns 429 after the threshold
- `POST /feedback` with valid body writes a JSONL line and returns ok
- `POST /feedback` with invalid rating returns 400

All 44 existing non-slow tests must stay green.

---

## Out of Scope (deliberate cuts)

- **Cross-question memory** — stateless for the beta; per-session memory is phase 2.
- **User accounts / auth** — anonymous access for the beta.
- **Database** — feedback to a flat file; no Postgres.
- **React / build pipeline** — plain static files.
- **Custom domain / HTTPS config** — use Railway's provided URL.
