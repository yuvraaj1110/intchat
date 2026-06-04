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
