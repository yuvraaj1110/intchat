# International Student RAG Chatbot

A free, retrieval-augmented chatbot answering F-1 / SEVIS / OPT / CPT, employment,
and student-life questions for international students in the U.S. — grounded in a
curated knowledge base with guardrails against hallucinated immigration advice.

## Quickstart

1. `pip install -r requirements.txt`
2. `cp .env.example .env` and add your free key from https://console.groq.com/keys
3. `python3 -m app.ingest --reset`   # build the vector store (one time, ~25s)
4. `python3 -m app.chat`             # start chatting

## Architecture

A modular LangChain pipeline. Each component is isolated behind a clean interface
so it can be swapped for a production-grade alternative in a single file.

```
question
  → hybrid retriever (semantic + exact-term filter)   app/retriever.py
  → guard-railed prompt (context-only + disclaimer)    app/prompts.py
  → Groq LLM (self-healing model selection + retry)    app/llm.py
  → windowed conversation memory                       app/chain.py
  → answer with topic citations
```

| Module | Responsibility |
|---|---|
| `app/config.py` | All settings: paths, model names, retrieval params, dedup IDs |
| `app/ingest.py` | Load → dedup → chunk → embed → store in ChromaDB |
| `app/retriever.py` | Hybrid semantic + keyword retrieval (Reciprocal Rank Fusion) |
| `app/prompts.py` | System prompt + RAG template (hallucination guardrails) |
| `app/llm.py` | Groq client; auto-selects a live model from a preference list |
| `app/chain.py` | Wires retriever + prompt + LLM + memory into `RAGChain.answer()` |
| `app/chat.py` | CLI REPL with input validation and startup checks |

## Knowledge base

121 deduplicated documents (general student life, F-1 employment/compliance, and
mentor-style Q&A). Dataset A and B overlap on OPT/CPT/STEM-OPT/SSN; the 8 less
detailed dataset-A versions are dropped at ingest in favor of dataset B's
structured format. Long documents are chunked (800-char window) for embedding.

## Tests

- `pytest -m "not slow"`  — fast unit tests (20)
- `pytest -m slow`        — tests that build a real vector store (4)
- `pytest`                — full suite (24)

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
| Memory | in-process window | Redis / Postgres (`app/chain.py`) |
| Frontend | CLI | FastAPI + web UI (new entry point) |

See `docs/superpowers/specs/2026-06-04-rag-chatbot-design.md` for the full design.
