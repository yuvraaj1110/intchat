# RAG Chatbot for International Students — Design Spec

**Date:** 2026-06-04
**Status:** Approved (design phase)
**Immediate goal:** A free, working, modular RAG chatbot (CLI first) that can be scaled to a production web service through single-file component swaps.

---

## 1. Overview

A retrieval-augmented chatbot that answers questions for international students in
the U.S. (F-1 visa status, SEVIS, OPT, CPT, employment rules, student life, and
university compliance). It answers **only** from a curated knowledge base of 121
normalized documents, with strict guardrails against hallucination given the
high-stakes immigration domain.

The first deliverable is a **command-line chatbot** that proves the retrieval +
generation pipeline works. A shareable web demo (FastAPI + simple HTML) is an
explicit **phase 2** follow-on, not part of this build. The architecture is
modular so each scaling step is a single-file change.

### Goals
- Working CLI chatbot answering from the knowledge base, today.
- Free to run: local embeddings + Groq free-tier LLM + embedded vector DB.
- Modular seams so every component can be swapped for a production-grade
  alternative without rewriting the rest.
- Resilient to API rate limits and common failure modes.

### Non-goals (this build)
- Web frontend / shareable demo URL (phase 2).
- Conversation persistence across restarts (phase 2).
- Automated retrieval evaluation harness (phase 2).
- Multi-user concurrency (requires Qdrant swap, phase 2).

---

## 2. Tech Stack

| Concern | Choice (now) | Rationale |
|---|---|---|
| Orchestration | LangChain (`>=0.3,<0.4`) | Standard RAG primitives; retriever/LLM/memory abstractions enable swaps |
| LLM | Groq, `llama-3.3-70b-versatile` | Free tier, very fast. **Model name must be verified live at build time** (Groq rotates/deprecates models) |
| Embeddings | `all-MiniLM-L6-v2` (local, 384-dim) | Free, no GPU, fast. Quality ceiling acceptable for demo |
| Vector DB | ChromaDB (embedded, persisted to disk) | No Docker, no server. Single-process limit is acceptable for CLI |
| Memory | `ConversationBufferWindowMemory` (k=10) | Bounds context window; in-process for CLI |
| Config | `python-dotenv` + `config.py` | All settings/keys in one place |

### Dependencies (`requirements.txt`, pinned to minor ranges)
```
langchain>=0.3.0,<0.4
langchain-groq>=0.2.0,<0.3
langchain-chroma>=0.2.0,<0.3
langchain-community>=0.3.0,<0.4
chromadb>=0.5.0,<0.6
sentence-transformers>=3.0.0,<4
python-dotenv>=1.0.0,<2
```

Versions are pinned to minor ranges to insulate against LangChain API churn.
LangChain-specific code is confined to thin wrapper modules (`retriever.py`,
`llm.py`, `chain.py`) so breakage is contained.

---

## 3. Project Structure

```
intchat/
├── datasets/                      # existing — raw + normalized data
│   └── normalized_dataset.json    # 129 docs from normalizer.py
├── app/                           # NEW — application code
│   ├── __init__.py
│   ├── config.py                  # settings, env vars, model names, paths
│   ├── ingest.py                  # load → dedup → chunk → embed → store
│   ├── retriever.py               # ChromaDB behind a hybrid retriever interface
│   ├── llm.py                     # Groq behind a LangChain LLM interface (+ retry)
│   ├── prompts.py                 # system prompt + RAG template
│   ├── chain.py                   # wires retriever + LLM + memory into a chain
│   └── chat.py                    # CLI entry point (chat loop)
├── normalizer.py                  # existing — data pipeline
├── requirements.txt               # NEW
├── .env                           # NEW — API keys (gitignored)
└── .env.example                   # NEW — template of required keys
```

Each module has one responsibility and communicates through a clean interface.
Swapping ChromaDB → Qdrant touches only `retriever.py`; Groq → another LLM
touches only `llm.py`; CLI → web adds a new entry point and leaves `chain.py`
untouched.

---

## 4. Ingestion Pipeline (`ingest.py`)

Flow: `normalized_dataset.json → dedup → chunk → embed → ChromaDB (persisted)`

### 4.1 Deduplication
Datasets A and B overlap on 4 topics (OPT, CPT, STEM OPT, SSN). Dataset B has the
richer structured format (eligibility, steps, rules, timelines, consequences), so
**dataset A's 8 docs covering those topics are dropped** during ingestion. This
prevents the retriever from returning conflicting, differently-worded docs for the
same question.

- Implementation: a hardcoded set of `(source, topic)` pairs to exclude, applied
  as a filter after loading. Corpus: 129 → **121 docs**.
- A's unique categories (SEVIS, I-20, I-901 fee, travel, programs of study,
  visas/status, life cycle/compliance, scholarships) are retained in full.

### 4.2 Chunking (for future larger documents)
Current docs average ~400 chars — already well-sized. Incoming large documents
(e.g. multi-page PDFs) are chunked with LangChain's `RecursiveCharacterTextSplitter`:
- chunk size 500 chars, overlap 50 chars
- split hierarchy: paragraph → sentence → word
- each chunk inherits parent metadata + a `chunk_index`

Documents already under the chunk size pass through unchanged.

### 4.3 Embedding & storage
- Embed each doc's `text` with `all-MiniLM-L6-v2`.
- Store in ChromaDB with full payload: `doc_id, text, category, topic, type,
  source, metadata`.
- Collection persisted to a local directory (e.g. `./chroma_db/`).
- Idempotent: re-running rebuilds the collection from scratch (`--reset`) or
  skips if already populated.

---

## 5. Retrieval (`retriever.py`) — Hybrid Search

Pure semantic search misses exact legal terms (`I-765`, `SEVP`, `EAD`, `I-901`).
The retriever combines two strategies and merges results.

```
query → ┌─ semantic search (embedding, top-8) ─┐
        │                                       ├─ merge + dedup → top-5 → context
        └─ metadata filter (known terms)      ─┘
```

1. **Semantic:** embed query → cosine similarity in ChromaDB → top-8 candidates.
2. **Metadata filter:** if the query contains known terms (form numbers like
   `I-765`/`I-20`, visa types like `F-1`/`J-1`, categories like `OPT`/`CPT`),
   apply a ChromaDB `where` filter to surface docs tagged with those terms.
3. **Merge:** combine via LangChain's `EnsembleRetriever` (Reciprocal Rank
   Fusion), dedup by `doc_id`, return top-5.

**Note on BM25:** true keyword/BM25 search is deferred. ChromaDB lacks native
full-text search; metadata filtering on known immigration terms covers the most
critical exact-match cases for v1. Full hybrid BM25 comes with the Qdrant swap.

---

## 6. LLM Layer (`llm.py`)

- `langchain-groq` wrapping `llama-3.3-70b-versatile` (model name from `config.py`,
  **verified against Groq's live model list at build time**).
- **Retry with exponential backoff:** 3 attempts on HTTP 429 / rate limit
  (delays 1s, 2s, 4s), then a graceful user-facing error.
- **Streaming** enabled for responsive CLI output.
- Swappable: any LangChain-compatible chat model by changing `config.py`.

---

## 7. Prompt Design (`prompts.py`)

The system prompt is the primary hallucination guardrail for this high-stakes
domain.

```
You are an assistant for international students in the United States.
You answer questions about F-1 visa status, SEVIS, OPT, CPT, employment
rules, student life, and university compliance — based ONLY on the
context documents provided below.

RULES:
1. ONLY answer from the provided context. If the context doesn't contain
   enough information, say: "I don't have specific information about that.
   Please check with your university's international student office (DSO)."
2. NEVER invent deadlines, form numbers, day counts, or eligibility rules.
3. After each answer, cite which topic(s) your answer came from.
4. Always end immigration-related answers with: "This is general guidance,
   not legal advice. Always confirm with your DSO or an immigration attorney."
5. Be warm, reassuring, and practical — many users are 17-18 year olds
   navigating this for the first time.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}
```

The "only from context" rule + mandatory disclaimer + citation requirement guard
against fabricated immigration advice. Tone matches the mentor-style dataset.

---

## 8. Chain & Memory (`chain.py`)

```
user message
   → ConversationBufferWindowMemory (last 10 exchanges)
   → hybrid retriever (top-5 docs)
   → RAG prompt (system + context + history + question)
   → Groq LLM (streaming)
   → response with citations + disclaimer
```

- Built with LangChain's retrieval chain (LCEL `create_retrieval_chain` or
  `ConversationalRetrievalChain`).
- Memory: `ConversationBufferWindowMemory(k=10)` — bounds the window so long
  conversations don't blow the context limit. In-process; swappable to
  Redis/Postgres for production.

---

## 9. CLI Entry Point (`chat.py`)

- REPL loop: read question → run chain → stream answer → repeat.
- On startup: validate `GROQ_API_KEY` present; check ChromaDB collection exists
  and is populated, otherwise prompt to run ingestion.
- Commands: `exit` to quit. Empty input ignored.

---

## 10. Error Handling & Resilience

| Failure | Handling |
|---|---|
| Groq rate limit (429) | Retry 3× with exponential backoff (1s/2s/4s), then graceful message |
| Groq service down | Catch connection errors; tell user to retry shortly |
| ChromaDB empty/missing | Detect on startup; instruct user to run `ingest.py` (or auto-run) |
| Query returns 0 results | Empty context → system prompt's "I don't have info" path |
| Malformed input | Reject empty queries; truncate input at 500 chars |
| Missing API key | Fail fast on startup with a clear `Set GROQ_API_KEY in .env` message |

---

## 11. Testing Strategy

- **Ingestion:** unit test that dedup drops exactly the 8 expected A docs and the
  corpus lands at 121; test that chunking splits an oversized doc and preserves
  metadata.
- **Retriever:** test that a query containing `I-765` surfaces the OPT/I-765 doc
  (exact-term case); test that a paraphrased query ("can I work after
  graduation?") surfaces OPT docs (semantic case).
- **Prompt/guardrail:** test that an off-topic query (e.g. "what's the weather?")
  yields the "I don't have specific information" fallback.
- **LLM retry:** test that a simulated 429 triggers backoff and eventually a
  graceful error rather than a crash.
- Manual smoke test of the full CLI against a handful of representative questions.

---

## 12. Production Migration Path

Every row is a single-file change enabled by the modular design.

| Component | Dev (now) | Production (later) | Change site |
|---|---|---|---|
| Vector DB | ChromaDB embedded | Qdrant Cloud / self-hosted | `retriever.py` |
| Embeddings | `all-MiniLM-L6-v2` | `bge-large-en-v1.5` / Voyage API | `config.py` |
| LLM | Groq free tier | Groq paid / OpenAI / Claude API | `llm.py` |
| Memory | in-process buffer | Redis / Postgres | `chain.py` |
| Frontend | CLI (`chat.py`) | FastAPI + React chat UI | new entry point |
| Search | semantic + metadata filter | hybrid semantic + BM25 | `retriever.py` (Qdrant) |
| Eval | manual | automated retrieval eval set | new module |

### Known scaling ceilings (acceptable for a free demo)
- **ChromaDB embedded** is ~single-process — first thing to swap under real
  concurrency.
- **Groq free tier** = 30 req/min, 14.4k/day — fine for a demo; paid key is a
  one-line change.
- **MiniLM embeddings** have a quality ceiling — swap when retrieval precision
  matters.

---

## 13. Open Items / Risks

- **Groq model name** (`llama-3.3-70b-versatile`) must be verified against the
  live model list at build time; Groq deprecates models periodically. This is the
  only known day-one breakage risk.
- Retrieval quality is unmeasured (no eval harness in v1) — validated manually.
- Conversation history is lost on restart (acceptable for CLI).
