# RAG Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a free, modular, CLI-based RAG chatbot that answers international-student questions strictly from a curated 121-document knowledge base.

**Architecture:** A LangChain pipeline — local sentence-transformers embeddings feed an embedded ChromaDB vector store; a hybrid retriever (semantic + metadata filter) selects context; a guard-railed prompt and Groq LLM generate cited answers. Each component lives in its own module behind a clean interface so it can be swapped for a production-grade alternative.

**Tech Stack:** Python 3.12, LangChain 0.3, `langchain-groq`, `langchain-chroma`, ChromaDB, `sentence-transformers` (`all-MiniLM-L6-v2`), Groq (`llama-3.3-70b-versatile`), pytest.

---

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned dependencies |
| `.env.example` | Template of required env vars |
| `app/__init__.py` | Marks `app` as a package |
| `app/config.py` | All settings: paths, model names, k values, loaded env vars |
| `app/ingest.py` | Load normalized docs → dedup → chunk → embed → store in ChromaDB |
| `app/retriever.py` | Build a hybrid retriever (semantic + metadata filter) over the store |
| `app/llm.py` | Construct the Groq chat model with retry/backoff |
| `app/prompts.py` | System prompt + RAG prompt template |
| `app/chain.py` | Wire retriever + prompt + LLM + memory into a callable chain |
| `app/chat.py` | CLI REPL entry point |
| `tests/conftest.py` | Shared fixtures (tiny in-memory document set, temp Chroma dir) |
| `tests/test_ingest.py` | Dedup + chunking behavior |
| `tests/test_retriever.py` | Semantic + exact-term retrieval |
| `tests/test_prompts.py` | Prompt assembly + guardrail text present |
| `tests/test_llm.py` | Retry/backoff on rate-limit |

Tasks are ordered so each produces self-contained, testable changes. Tests use a tiny fixture corpus (no network, no model download where avoidable) except where a real embedding is required; those are marked.

---

### Task 1: Project scaffolding & dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```
langchain>=0.3.0,<0.4
langchain-groq>=0.2.0,<0.3
langchain-chroma>=0.2.0,<0.3
langchain-huggingface>=0.1.0,<0.3
langchain-community>=0.3.0,<0.4
chromadb>=0.5.0,<0.6
sentence-transformers>=3.0.0,<4
python-dotenv>=1.0.0,<2
pytest>=8.0.0
```

- [ ] **Step 2: Create `.env.example`**

```
# Get a free key at https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here
```

- [ ] **Step 3: Create empty package markers**

Create `app/__init__.py` with a single line:

```python
"""International student RAG chatbot application package."""
```

Create `tests/__init__.py` as an empty file (0 bytes).

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: completes; `chromadb`, `langchain-groq`, `langchain-chroma`, `langchain-huggingface` newly installed (langchain, sentence-transformers, pytest already present).

- [ ] **Step 5: Verify imports**

Run: `python3 -c "import chromadb, langchain_groq, langchain_chroma, langchain_huggingface, sentence_transformers; print('ok')"`
Expected: prints `ok`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example app/__init__.py tests/__init__.py
git commit -m "chore: scaffold app package and pin dependencies"
```

---

### Task 2: Configuration module

**Files:**
- Create: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from app import config


def test_config_has_expected_constants():
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.GROQ_MODEL == "llama-3.3-70b-versatile"
    assert config.TOP_K == 5
    assert config.SEMANTIC_K == 8
    assert config.MEMORY_WINDOW == 10
    assert config.COLLECTION_NAME == "intchat_knowledge"
    # Paths are pathlib.Path objects
    assert config.NORMALIZED_DATASET.name == "normalized_dataset.json"
    assert config.CHROMA_DIR.name == "chroma_db"


def test_dedup_exclusions_are_the_eight_overlap_docs():
    assert config.DEDUP_EXCLUDE_IDS == {
        "dataset_A__cpt_guidance__0",
        "dataset_A__cpt_guidance__1",
        "dataset_A__opt_guidance__0",
        "dataset_A__opt_guidance__1",
        "dataset_A__stem_opt__0",
        "dataset_A__stem_opt__1",
        "dataset_A__ssn_guidance__0",
        "dataset_A__ssn_guidance__1",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError` (config not yet defined)

- [ ] **Step 3: Write `app/config.py`**

```python
"""Central configuration for the RAG chatbot.

All tunable settings, model names, and paths live here so that swapping a
component (vector DB, embedding model, LLM) is a single-file change.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
NORMALIZED_DATASET = BASE_DIR / "datasets" / "normalized_dataset.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── Vector store ─────────────────────────────────────────────────────────
COLLECTION_NAME = "intchat_knowledge"

# ── Embeddings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking (for future large documents) ────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Retrieval ────────────────────────────────────────────────────────────
SEMANTIC_K = 8   # candidates pulled by semantic search before merge
TOP_K = 5        # final documents passed to the LLM as context

# ── LLM ──────────────────────────────────────────────────────────────────
# NOTE: verify this model name against Groq's live model list at build time;
# Groq deprecates models periodically. See https://console.groq.com/docs/models
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MAX_RETRIES = 3
LLM_TEMPERATURE = 0.2

# ── Memory ───────────────────────────────────────────────────────────────
MEMORY_WINDOW = 10  # number of past exchanges retained

# ── Deduplication ────────────────────────────────────────────────────────
# Dataset A and B both cover OPT / CPT / STEM-OPT / SSN. Dataset B has the
# richer structured format, so these 8 dataset_A docs are dropped at ingest.
DEDUP_EXCLUDE_IDS = {
    "dataset_A__cpt_guidance__0",
    "dataset_A__cpt_guidance__1",
    "dataset_A__opt_guidance__0",
    "dataset_A__opt_guidance__1",
    "dataset_A__stem_opt__0",
    "dataset_A__stem_opt__1",
    "dataset_A__ssn_guidance__0",
    "dataset_A__ssn_guidance__1",
}

# Known immigration terms used by the metadata-filter half of hybrid retrieval.
KNOWN_TERMS = [
    "I-765", "I-20", "I-983", "I-901", "I-17",
    "F-1", "M-1", "J-1",
    "OPT", "CPT", "STEM", "SEVIS", "SEVP", "EAD", "SSN", "DSO",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add central config module"
```

---

### Task 3: Shared test fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures.

`sample_docs` is a tiny corpus that mirrors the real normalized schema. It is
deliberately small so embedding-backed tests stay fast.
"""

import pytest


@pytest.fixture
def sample_docs():
    """Six docs: includes one A/B overlap pair and one exact-term doc."""
    return [
        {
            "id": "dataset_A__opt_guidance__0",
            "text": "OPT allows F-1 students to work in their field for up to 12 months.",
            "category": "Practical Training and Employment",
            "topic": "Optional Practical Training (OPT)",
            "type": "paragraph",
            "source": "dataset_A",
            "metadata": {},
        },
        {
            "id": "dataset_B__opt_optional_practical_training__summary",
            "text": "OPT allows F-1 students to engage in temporary employment "
                    "related to their major. File Form I-765 to apply.",
            "category": "OPT",
            "topic": "Optional Practical Training (OPT)",
            "type": "summary",
            "source": "dataset_B",
            "metadata": {"required_forms": ["I-20", "I-765"]},
        },
        {
            "id": "dataset_A__sevis_overview__0",
            "text": "SEVIS is the system DHS uses to track F-1 and M-1 students.",
            "category": "SEVIS System Overview",
            "topic": "SEVIS Purpose and Management",
            "type": "paragraph",
            "source": "dataset_A",
            "metadata": {},
        },
        {
            "id": "mentorstyle__p1_meaning_001",
            "text": "Q: What does it mean to be an international student?\n"
                    "A: You are a guest student here to study on a visa.",
            "category": "What Being an International Student Really Means",
            "topic": "What does it mean to be an international student?",
            "type": "qa",
            "source": "mentorstyle",
            "metadata": {},
        },
        {
            "id": "dataset_B__ssn__summary",
            "text": "F-1 students with authorized employment may apply for an SSN.",
            "category": "SSN",
            "topic": "Social Security Number (SSN)",
            "type": "summary",
            "source": "dataset_B",
            "metadata": {},
        },
        {
            "id": "dataset_A__travel_entry__0",
            "text": "Enter the U.S. no earlier than 30 days before your program start.",
            "category": "Travel and Entry",
            "topic": "Entering the United States",
            "type": "paragraph",
            "source": "dataset_A",
            "metadata": {},
        },
    ]
```

- [ ] **Step 2: Verify the fixture loads**

Run: `python3 -c "import tests.conftest; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared sample_docs fixture"
```

---

### Task 4: Ingestion — dedup & chunk (pure functions)

**Files:**
- Create: `app/ingest.py`
- Test: `tests/test_ingest.py`

This task builds the two pure transforms (`deduplicate`, `chunk_docs`). The
ChromaDB-writing part comes in Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
from app import ingest


def test_deduplicate_drops_excluded_ids(sample_docs):
    result = ingest.deduplicate(sample_docs)
    ids = {d["id"] for d in result}
    # The A-side OPT doc is in DEDUP_EXCLUDE_IDS and must be gone
    assert "dataset_A__opt_guidance__0" not in ids
    # The B-side OPT doc stays
    assert "dataset_B__opt_optional_practical_training__summary" in ids
    # Non-overlapping A docs stay
    assert "dataset_A__sevis_overview__0" in ids


def test_deduplicate_on_full_dataset_yields_121():
    import json
    from app import config
    docs = json.load(open(config.NORMALIZED_DATASET))
    assert len(docs) == 129
    assert len(ingest.deduplicate(docs)) == 121


def test_chunk_docs_passes_through_small_docs(sample_docs):
    chunks = ingest.chunk_docs(sample_docs)
    # All sample docs are short, so count is unchanged
    assert len(chunks) == len(sample_docs)
    # Schema preserved
    assert set(chunks[0].keys()) >= {"id", "text", "category", "source"}


def test_chunk_docs_splits_a_long_doc():
    long_doc = [{
        "id": "big__0",
        "text": "word " * 400,  # ~2000 chars, exceeds CHUNK_SIZE=500
        "category": "C", "topic": "T", "type": "paragraph",
        "source": "big", "metadata": {"a": 1},
    }]
    chunks = ingest.chunk_docs(long_doc)
    assert len(chunks) > 1
    # Each chunk keeps parent metadata and gets a chunk_index
    assert all(c["category"] == "C" for c in chunks)
    assert all("chunk_index" in c for c in chunks)
    # Chunk ids are unique and derived from the parent id
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert all(c["id"].startswith("big__0") for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingest'`

- [ ] **Step 3: Write the pure transforms in `app/ingest.py`**

```python
"""Ingestion pipeline: load → dedup → chunk → embed → store in ChromaDB.

This module exposes pure transforms (`deduplicate`, `chunk_docs`) plus a
`build_store` function that writes to ChromaDB, and a CLI entry point.
"""

import json
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config


def load_docs() -> list[dict[str, Any]]:
    """Load the normalized dataset from disk."""
    with open(config.NORMALIZED_DATASET, encoding="utf-8") as f:
        return json.load(f)


def deduplicate(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop dataset_A docs superseded by dataset_B's richer versions."""
    return [d for d in docs if d["id"] not in config.DEDUP_EXCLUDE_IDS]


def chunk_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split any doc whose text exceeds CHUNK_SIZE; pass short docs through.

    Each emitted chunk keeps the parent's metadata and gains a `chunk_index`
    and a unique id of the form `<parent_id>__c<n>` (single-chunk docs keep
    the parent id unchanged).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    out: list[dict[str, Any]] = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        if len(pieces) <= 1:
            single = {**doc, "chunk_index": 0}
            out.append(single)
            continue
        for i, piece in enumerate(pieces):
            out.append({
                **doc,
                "id": f"{doc['id']}__c{i}",
                "text": piece,
                "chunk_index": i,
            })
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/ingest.py tests/test_ingest.py
git commit -m "feat: add ingestion dedup and chunking transforms"
```

---

### Task 5: Ingestion — embed & store in ChromaDB

**Files:**
- Modify: `app/ingest.py` (append `build_store` + CLI `main`)
- Test: `tests/test_ingest_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_store.py
import pytest

from app import ingest


@pytest.mark.slow
def test_build_store_creates_searchable_collection(tmp_path, sample_docs):
    store = ingest.build_store(
        sample_docs, persist_dir=str(tmp_path), reset=True
    )
    # Five sample docs survive dedup (one A-OPT doc is excluded)
    results = store.similarity_search("how does OPT work", k=3)
    assert len(results) >= 1
    # The B-side OPT doc should be the top hit, not the dropped A one
    joined = " ".join(r.page_content for r in results)
    assert "I-765" in joined or "temporary employment" in joined
    # Payload metadata is preserved
    assert results[0].metadata.get("source") in {"dataset_A", "dataset_B", "mentorstyle"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest_store.py -v`
Expected: FAIL with `AttributeError: module 'app.ingest' has no attribute 'build_store'`

- [ ] **Step 3: Append `build_store` and `main` to `app/ingest.py`**

Add these imports at the top of `app/ingest.py` (below the existing imports):

```python
import argparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
```

Then append to the end of `app/ingest.py`:

```python
def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def _to_documents(docs: list[dict[str, Any]]) -> list[Document]:
    """Convert our dicts to LangChain Documents with flattened metadata.

    Chroma metadata values must be str/int/float/bool, so the nested
    `metadata` dict is dropped from the stored metadata (its useful fields —
    category, topic, type, source — are already top-level).
    """
    documents = []
    for d in docs:
        documents.append(Document(
            page_content=d["text"],
            metadata={
                "doc_id": d["id"],
                "category": d.get("category", ""),
                "topic": d.get("topic", ""),
                "type": d.get("type", ""),
                "source": d.get("source", ""),
                "chunk_index": d.get("chunk_index", 0),
            },
        ))
    return documents


def build_store(docs, persist_dir=None, reset=False):
    """Dedup → chunk → embed → write a persisted Chroma collection.

    `docs` is the raw list of normalized dicts. Returns the Chroma store.
    """
    persist_dir = persist_dir or str(config.CHROMA_DIR)
    prepared = chunk_docs(deduplicate(docs))
    documents = _to_documents(prepared)

    if reset:
        # Fresh build: Chroma.from_documents overwrites the collection contents.
        store = Chroma.from_documents(
            documents=documents,
            embedding=_get_embeddings(),
            collection_name=config.COLLECTION_NAME,
            persist_directory=persist_dir,
        )
    else:
        store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=persist_dir,
        )
        store.add_documents(documents)
    return store


def main():
    parser = argparse.ArgumentParser(description="Ingest knowledge base into ChromaDB")
    parser.add_argument("--reset", action="store_true",
                        help="Rebuild the collection from scratch")
    args = parser.parse_args()

    docs = load_docs()
    print(f"Loaded {len(docs)} docs")
    store = build_store(docs, reset=args.reset)
    count = store._collection.count()
    print(f"✅ Ingested into '{config.COLLECTION_NAME}' — {count} chunks stored")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the `slow` marker and run the test**

Create `pytest.ini` in the repo root:

```ini
[pytest]
markers =
    slow: tests that download models or build a real vector store
```

Run: `pytest tests/test_ingest_store.py -v -m slow`
Expected: PASS (1 passed) — first run downloads the embedding model (~90MB), so allow time.

- [ ] **Step 5: Build the real store end to end**

Run: `python3 -m app.ingest --reset`
Expected: prints `Loaded 129 docs` then `✅ Ingested into 'intchat_knowledge' — 121 chunks stored`

- [ ] **Step 6: Commit**

```bash
git add app/ingest.py tests/test_ingest_store.py pytest.ini
git commit -m "feat: embed and store knowledge base in ChromaDB"
```

---

### Task 6: Hybrid retriever

**Files:**
- Create: `app/retriever.py`
- Test: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
import pytest

from app import ingest, retriever


@pytest.fixture
def built_store(tmp_path, sample_docs):
    return ingest.build_store(sample_docs, persist_dir=str(tmp_path), reset=True)


def test_extract_known_terms_finds_form_numbers():
    terms = retriever.extract_known_terms("Do I need form I-765 for OPT?")
    assert "I-765" in terms
    assert "OPT" in terms


def test_extract_known_terms_empty_when_none_present():
    assert retriever.extract_known_terms("how do I make friends on campus?") == []


@pytest.mark.slow
def test_retrieve_semantic_paraphrase(built_store):
    r = retriever.build_retriever(built_store)
    docs = r.invoke("can I work in my field after graduation?")
    assert any("OPT" in d.page_content or d.metadata.get("category") in {"OPT", "Practical Training and Employment"}
               for d in docs)


@pytest.mark.slow
def test_retrieve_exact_term_surfaces_i765_doc(built_store):
    r = retriever.build_retriever(built_store)
    docs = r.invoke("which form is I-765 used for?")
    assert any("I-765" in d.page_content for d in docs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retriever'`

- [ ] **Step 3: Write `app/retriever.py`**

```python
"""Hybrid retriever: semantic search merged with a metadata/keyword filter.

Pure semantic search can miss exact legal terms (I-765, SEVP, EAD). This
retriever runs a semantic search and, when the query contains known terms,
also runs a keyword search over document text, then merges the two rankings.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app import config


def extract_known_terms(query: str) -> list[str]:
    """Return known immigration terms present in the query (case-insensitive)."""
    upper = query.upper()
    return [t for t in config.KNOWN_TERMS if t.upper() in upper]


class HybridRetriever(BaseRetriever):
    """Merge semantic hits with keyword hits via Reciprocal Rank Fusion."""

    store: Any = Field(...)
    semantic_k: int = Field(default=config.SEMANTIC_K)
    top_k: int = Field(default=config.TOP_K)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        semantic = self.store.similarity_search(query, k=self.semantic_k)

        keyword: list[Document] = []
        terms = extract_known_terms(query)
        if terms:
            # Pull a wide semantic net, then keep docs whose text contains a term.
            candidates = self.store.similarity_search(query, k=self.semantic_k * 2)
            keyword = [
                d for d in candidates
                if any(t.upper() in d.page_content.upper() for t in terms)
            ]

        return self._fuse(semantic, keyword)[: self.top_k]

    def _fuse(self, *rankings: list[Document], k: int = 60) -> list[Document]:
        """Reciprocal Rank Fusion across multiple ranked lists."""
        scores: dict[str, float] = {}
        by_id: dict[str, Document] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                key = doc.metadata.get("doc_id", doc.page_content[:50])
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                by_id[key] = doc
        ordered = sorted(scores, key=scores.get, reverse=True)
        return [by_id[key] for key in ordered]


def build_retriever(store) -> HybridRetriever:
    """Construct a HybridRetriever over an existing Chroma store."""
    return HybridRetriever(store=store)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS (4 passed; 2 are slow and build a real store)

- [ ] **Step 5: Commit**

```bash
git add app/retriever.py tests/test_retriever.py
git commit -m "feat: add hybrid semantic + keyword retriever"
```

---

### Task 7: Prompt templates

**Files:**
- Create: `app/prompts.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
from app import prompts


def test_system_prompt_has_guardrails():
    text = prompts.SYSTEM_PROMPT
    assert "ONLY" in text
    assert "DSO" in text
    assert "not legal advice" in text


def test_build_prompt_includes_context_and_question():
    msgs = prompts.build_prompt(
        context="OPT allows F-1 students to work.",
        chat_history="",
        question="What is OPT?",
    )
    rendered = "\n".join(m.content for m in msgs)
    assert "OPT allows F-1 students to work." in rendered
    assert "What is OPT?" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.prompts'`

- [ ] **Step 3: Write `app/prompts.py`**

```python
"""System prompt and RAG prompt assembly.

The system prompt is the primary hallucination guardrail for this high-stakes
immigration domain: answer only from context, never invent rules, cite topics,
and append a legal disclaimer.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an assistant for international students in the United States.
You answer questions about F-1 visa status, SEVIS, OPT, CPT, employment rules,
student life, and university compliance — based ONLY on the context documents
provided below.

RULES:
1. ONLY answer from the provided context. If the context doesn't contain enough
   information, say: "I don't have specific information about that. Please check
   with your university's international student office (DSO)."
2. NEVER invent deadlines, form numbers, day counts, or eligibility rules.
3. After each answer, cite which topic(s) your answer came from.
4. Always end immigration-related answers with: "This is general guidance, not
   legal advice. Always confirm with your DSO or an immigration attorney."
5. Be warm, reassuring, and practical — many users are 17-18 year olds
   navigating this for the first time."""

_HUMAN_TEMPLATE = """CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION:
{question}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", _HUMAN_TEMPLATE),
])


def build_prompt(context: str, chat_history: str, question: str):
    """Return the rendered list of chat messages for a single turn."""
    return PROMPT.format_messages(
        context=context, chat_history=chat_history, question=question
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/prompts.py tests/test_prompts.py
git commit -m "feat: add guard-railed prompt templates"
```

---

### Task 8: LLM client with retry/backoff

**Files:**
- Create: `app/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import pytest

from app import llm


def test_build_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.build_llm()


def test_build_llm_uses_configured_model(monkeypatch):
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", "test-key")
    model = llm.build_llm()
    assert model.model_name == llm.config.GROQ_MODEL
    assert model.max_retries == llm.config.LLM_MAX_RETRIES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.llm'`

- [ ] **Step 3: Write `app/llm.py`**

```python
"""Groq chat model construction with built-in retry/backoff.

`ChatGroq` retries transient errors (including HTTP 429 rate limits) up to
`max_retries` times with exponential backoff, satisfying the resilience
requirement without custom retry code.
"""

from langchain_groq import ChatGroq

from app import config


def build_llm(streaming: bool = True) -> ChatGroq:
    """Construct the Groq chat model. Raises if the API key is missing."""
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys"
        )
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_retries=config.LLM_MAX_RETRIES,
        streaming=streaming,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (2 passed)

> If `model.model_name` raises an attribute error on the installed
> `langchain-groq` version, use `model.model` instead and update the test
> assertion to match — both refer to the same configured model string.

- [ ] **Step 5: Commit**

```bash
git add app/llm.py tests/test_llm.py
git commit -m "feat: add Groq LLM client with retry/backoff"
```

---

### Task 9: RAG chain with memory

**Files:**
- Create: `app/chain.py`
- Test: `tests/test_chain.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.chain'`

- [ ] **Step 3: Write `app/chain.py`**

```python
"""Assemble retriever + prompt + LLM + memory into a single answer() call.

A lightweight `ConversationWindow` holds the last N exchanges (bounding the
context window). `RAGChain.answer` retrieves context, renders the prompt, and
invokes the LLM. Memory is in-process; swap for Redis/Postgres in production.
"""

from collections import deque

from app import config, prompts
from app.llm import build_llm
from app.retriever import build_retriever


def format_context(docs) -> str:
    """Render retrieved docs as a context block with topic labels."""
    blocks = []
    for d in docs:
        topic = d.metadata.get("topic", "")
        label = f"[{topic}] " if topic else ""
        blocks.append(f"{label}{d.page_content}")
    return "\n\n".join(blocks)


class ConversationWindow:
    """Keep the most recent `max_pairs` (question, answer) exchanges."""

    def __init__(self, max_pairs: int = config.MEMORY_WINDOW):
        self._pairs: deque = deque(maxlen=max_pairs)

    def add(self, question: str, answer: str) -> None:
        self._pairs.append((question, answer))

    def render(self) -> str:
        return "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in self._pairs
        )


class RAGChain:
    """End-to-end retrieval-augmented chat."""

    def __init__(self, store, llm=None):
        self.retriever = build_retriever(store)
        self.llm = llm or build_llm(streaming=False)
        self.memory = ConversationWindow()

    def answer(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = format_context(docs)
        messages = prompts.build_prompt(
            context=context,
            chat_history=self.memory.render(),
            question=question,
        )
        response = self.llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        self.memory.add(question, text)
        return text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chain.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/chain.py tests/test_chain.py
git commit -m "feat: assemble RAG chain with windowed memory"
```

---

### Task 10: CLI entry point

**Files:**
- Create: `app/chat.py`
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat.py
from app import chat


def test_validate_query_rejects_empty():
    assert chat.validate_query("") is None
    assert chat.validate_query("   ") is None


def test_validate_query_truncates_long_input():
    long = "a" * 1000
    assert len(chat.validate_query(long)) == 500


def test_validate_query_passes_normal_input():
    assert chat.validate_query("What is OPT?") == "What is OPT?"


def test_load_store_errors_when_missing(tmp_path, monkeypatch):
    # Point at an empty dir → no collection → clear error
    monkeypatch.setattr(chat.config, "CHROMA_DIR", tmp_path / "nope")
    import pytest
    with pytest.raises(SystemExit):
        chat.load_store()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.chat'`

- [ ] **Step 3: Write `app/chat.py`**

```python
"""CLI REPL for the international-student RAG chatbot.

Startup validates the API key and that a populated Chroma collection exists,
then loops: read question → validate → answer → print.
"""

import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app import config
from app.chain import RAGChain

MAX_QUERY_LEN = 500


def validate_query(raw: str) -> str | None:
    """Return a cleaned query, or None if it is empty."""
    cleaned = raw.strip()
    if not cleaned:
        return None
    return cleaned[:MAX_QUERY_LEN]


def load_store() -> Chroma:
    """Open the persisted Chroma collection, or exit with guidance."""
    if not config.CHROMA_DIR.exists():
        sys.exit(
            "No knowledge base found. Run:  python3 -m app.ingest --reset"
        )
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL),
        persist_directory=str(config.CHROMA_DIR),
    )
    if store._collection.count() == 0:
        sys.exit(
            "Knowledge base is empty. Run:  python3 -m app.ingest --reset"
        )
    return store


def main() -> None:
    if not config.GROQ_API_KEY:
        sys.exit(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )
    store = load_store()
    rag = RAGChain(store)

    print("International Student Assistant — ask a question (type 'exit' to quit)")
    while True:
        try:
            raw = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw.strip().lower() == "exit":
            break
        query = validate_query(raw)
        if query is None:
            continue
        try:
            answer = rag.answer(query)
        except Exception as exc:  # noqa: BLE001 - surface a friendly message
            print(f"\nSorry, something went wrong: {exc}\nPlease try again in a moment.")
            continue
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chat.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/chat.py tests/test_chat.py
git commit -m "feat: add CLI chat entry point"
```

---

### Task 11: Full suite + live smoke test

**Files:**
- Modify: `README` section (create `README.md` if absent)

- [ ] **Step 1: Run the full test suite (excluding slow)**

Run: `pytest -v -m "not slow"`
Expected: all fast tests PASS.

- [ ] **Step 2: Run the slow suite**

Run: `pytest -v -m slow`
Expected: ingestion + retriever slow tests PASS (builds real stores).

- [ ] **Step 3: Ensure the real knowledge base is built**

Run: `python3 -m app.ingest --reset`
Expected: `✅ Ingested into 'intchat_knowledge' — 121 chunks stored`

- [ ] **Step 4: Live smoke test (requires a real GROQ_API_KEY in `.env`)**

Run: `python3 -m app.chat`
Then type: `Do I need to file form I-765 for OPT?`
Expected: an answer that references OPT and I-765, cites the topic, and ends with the legal-advice disclaimer. Type `exit` to quit.

> If the first call returns a model error, the Groq model name has likely been
> rotated. Check https://console.groq.com/docs/models and update `GROQ_MODEL`
> in `app/config.py`.

- [ ] **Step 5: Write `README.md` quickstart**

```markdown
# International Student RAG Chatbot

A free, retrieval-augmented chatbot answering F-1 / SEVIS / OPT / CPT and
student-life questions from a curated knowledge base.

## Quickstart

1. `pip install -r requirements.txt`
2. `cp .env.example .env` and add your free key from https://console.groq.com/keys
3. `python3 -m app.ingest --reset`   # build the vector store (one time)
4. `python3 -m app.chat`             # start chatting

## Tests

- `pytest -m "not slow"`  — fast unit tests
- `pytest -m slow`        — tests that build a real vector store

## Architecture

See `docs/superpowers/specs/2026-06-04-rag-chatbot-design.md`. Each component
(vector DB, embeddings, LLM, memory, frontend) is isolated for single-file
swaps toward a production deployment.
```

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: add quickstart README"
```

---

## Self-Review Notes

**Spec coverage:**
- Dedup (129→121) → Task 4 (+ verified live in Task 5).
- Chunking → Task 4.
- Embedding & ChromaDB storage → Task 5.
- Hybrid retrieval (semantic + metadata/known-term filter) → Task 6.
- Guard-railed prompt (context-only, citations, disclaimer) → Task 7.
- Groq LLM + retry/backoff → Task 8 (via `ChatGroq.max_retries`).
- Conversation window memory → Task 9.
- CLI + startup validation + error handling → Task 10.
- Config-driven swaps + pinned deps → Tasks 1–2.
- Testing strategy → Tasks 4–10 tests; full run in Task 11.

**Known build-time check:** Groq model name (`llama-3.3-70b-versatile`) must be
verified against the live model list — called out in `config.py` and Task 11.
