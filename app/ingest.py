"""Ingestion pipeline: load → dedup → chunk → embed → store in ChromaDB.

This module exposes pure transforms (`deduplicate`, `chunk_docs`) plus a
`build_store` function that writes to ChromaDB, and a CLI entry point.
"""

import json
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config

import argparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


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
        meta = {
            "doc_id": d["id"],
            "category": d.get("category", ""),
            "topic": d.get("topic", ""),
            "type": d.get("type", ""),
            "source": d.get("source", ""),
            "chunk_index": d.get("chunk_index", 0),
        }
        # Provenance fields from web/PDF sources (Chroma needs flat scalars)
        inner_meta = d.get("metadata", {})
        if inner_meta.get("source_url"):
            meta["source_url"] = inner_meta["source_url"]
        if inner_meta.get("source_name"):
            meta["source_name"] = inner_meta["source_name"]
        if inner_meta.get("fetched_at"):
            meta["fetched_at"] = inner_meta["fetched_at"]
        documents.append(Document(
            page_content=d["text"],
            metadata=meta,
        ))
    return documents


def build_store(docs, persist_dir=None, reset=False):
    """Dedup → chunk → embed → write a persisted Chroma collection.

    `docs` is the raw list of normalized dicts. Returns the Chroma store.
    When `reset` is True, any existing collection is cleared first so repeated
    rebuilds are idempotent (they do not accumulate duplicates).
    """
    persist_dir = persist_dir or str(config.CHROMA_DIR)
    prepared = chunk_docs(deduplicate(docs))
    documents = _to_documents(prepared)
    embeddings = _get_embeddings()

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    if reset:
        # Clear any existing data so a rebuild is idempotent.
        store.reset_collection()
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
