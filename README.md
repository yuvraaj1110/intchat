---
title: International Student Assistant
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# International Student RAG Chatbot

A free, retrieval-augmented chatbot answering F-1 / SEVIS / OPT / CPT, employment,
and student-life questions for international students in the U.S. — grounded in a
knowledge base sourced from USCIS, DHS, State Dept, Purdue ISS, and curated
datasets, with guardrails against hallucinated immigration advice and full
provenance (source URL + fetch date) in every answer.

## Quickstart

1. `pip install -r requirements.txt`
2. `cp .env.example .env` and add your free key from https://console.groq.com/keys
3. `python3 -m app.build_kb --reset`  # fetch sources + build vector store (~60s)
4. `python3 -m app.chat`              # start chatting (CLI)

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
feedback is logged to `feedback.jsonl` on the server. Answers stream token by
token (SSE) and show source URLs + fetch dates in a sidebar.

## Architecture

A modular LangChain pipeline. Each component is isolated behind a clean interface
so it can be swapped for a production-grade alternative in a single file.

```
question
  → hybrid retriever (semantic + exact-term filter)   app/retriever.py
  → guard-railed prompt (context-only + disclaimer)    app/prompts.py
  → Groq LLM (self-healing model selection + retry)    app/llm.py
  → windowed conversation memory                       app/chain.py
  → answer with source citations (URL + date)
```

| Module | Responsibility |
|---|---|
| `app/config.py` | All settings: paths, model names, retrieval params, dedup IDs |
| `app/build_kb.py` | Orchestrator: fetch all sources → parse → normalize → ingest |
| `app/fetch_html.py` | Download web pages with browser-like headers |
| `app/fetch_pdf.py` | Extract text from local PDF files |
| `app/parse_html.py` | Strip boilerplate (nav, footer, ads) via trafilatura |
| `app/web_normalizer.py` | Convert cleaned text to normalized schema + provenance |
| `app/ingest.py` | Dedup → chunk → embed → store in ChromaDB |
| `app/retriever.py` | Hybrid semantic + keyword retrieval (Reciprocal Rank Fusion) |
| `app/prompts.py` | System prompt + RAG template (hallucination guardrails) |
| `app/llm.py` | Groq client; auto-selects a live model from a preference list |
| `app/chain.py` | Wires retriever + prompt + LLM + memory into `RAGChain.answer()` |
| `app/chat.py` | CLI REPL with input validation and startup checks |
| `app/server.py` | FastAPI web server: SSE streaming, rate limiting, feedback |
| `app/static/` | Q&A web UI (HTML/CSS/JS) with source sidebar |

## Knowledge base

219 chunks from 142 documents sourced from:
- **Government sites** — USCIS, DHS Study in the States, State Dept (live-fetched)
- **Purdue ISS** — Employment, CPT, OPT, travel, new students (live-fetched)
- **Curated datasets** — Hand-written student life, compliance, and Q&A (JSON)

To add more sources, edit `sources.yaml` and re-run `python3 -m app.build_kb --reset`.
Answers cite source URLs and retrieval dates for verifiability.

## Tests

- `pytest -m "not slow"`  — fast unit tests (54)
- `pytest -m slow`        — tests that build a real vector store (4)
- `pytest`                — full suite (58)

## Notes on the Groq model

Groq hosts third-party open models and retires them periodically, so no single
model name stays valid forever. `app/llm.py` queries Groq's live model list at
startup and selects the first available entry from
`config.GROQ_MODEL_PREFERENCES`. It only fails if Groq has retired every preferred
model — then it prints the available list so you can update the preferences.

## Scaling to production

Every component is a single-file swap:

| Component | Now (free demo) | Production |
|---|---|---|
| Vector DB | ChromaDB (embedded) | Qdrant (`app/retriever.py`) |
| Embeddings | `all-MiniLM-L6-v2` | `bge-large-en-v1.5` / API (`app/config.py`) |
| LLM | Groq free tier | Groq paid / OpenAI / Claude (`app/llm.py`) |
| Memory | in-process window (CLI) / stateless (web) | Redis / Postgres (`app/chain.py`) |
| Frontend | CLI + FastAPI web UI | + per-session memory, auth, custom domain |
| Hosting | Railway free tier (Docker) | Fly.io / VPS with autoscaling |
| Rate limiting | in-memory per-IP (`slowapi`) | Redis-backed distributed limits |

See `docs/superpowers/specs/2026-06-04-rag-chatbot-design.md` for the full design.
