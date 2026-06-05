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
NORMALIZED_ALL = BASE_DIR / "datasets" / "normalized_all.json"
SOURCES_YAML = BASE_DIR / "sources.yaml"
PDFS_DIR = BASE_DIR / "datasets" / "pdfs"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ── Vector store ─────────────────────────────────────────────────────────
COLLECTION_NAME = "intchat_knowledge"

# ── Embeddings ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking (for future large documents) ────────────────────────────────
# CHUNK_SIZE is tuned to the embedding model's window. all-MiniLM-L6-v2
# truncates input past ~256 tokens (~1000 chars), so 800 chars keeps every
# chunk safely inside the window (no silent truncation) while leaving the
# curated docs — already coherent units, max 1067 chars — almost entirely
# intact. Only genuinely oversized docs (e.g. future multi-page PDFs) split.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50

# ── Retrieval ────────────────────────────────────────────────────────────
SEMANTIC_K = 8   # candidates pulled by semantic search before merge
TOP_K = 5        # final documents passed to the LLM as context

# ── Input handling ───────────────────────────────────────────────────────
MAX_QUERY_LEN = 500  # user queries are truncated to this many characters

# ── LLM ──────────────────────────────────────────────────────────────────
# Groq hosts third-party open models and deprecates them periodically, so no
# single model name is permanently valid. Instead of hardcoding one, we keep a
# preference list and select the first one Groq actually serves at startup
# (see app/llm.py). Update/reorder this list as Groq's catalog changes.
# See https://console.groq.com/docs/models
GROQ_MODEL_PREFERENCES = [
    "llama-3.3-70b-versatile",   # preferred: strong reasoning
    "llama-3.1-8b-instant",      # fallback: faster, smaller
    "llama3-70b-8192",           # fallback: older 70B
]
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
    "I-765", "I-20", "I-983", "I-901", "I-17", "I-94",
    "F-1", "M-1", "J-1", "H-1B",
    "OPT", "CPT", "STEM", "SEVIS", "SEVP", "EAD", "SSN", "DSO",
    "FICA", "1040-NR", "W-4", "W-8BEN", "ITIN",
    "Glacier", "Sprintax",
]
