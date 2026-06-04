"""
Ingest the normalized dataset into Qdrant for semantic search.

Reads the flat document list produced by normalizer.py and:
  1. Embeds each document's text with sentence-transformers
  2. Upserts into a Qdrant collection with full payload metadata

Usage:
    python ingest_qdrant.py                     # defaults
    python ingest_qdrant.py --reset             # drop + recreate collection first
    python ingest_qdrant.py --file ../normalized_dataset.json
"""

import json
import hashlib
import argparse
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
VECTOR_SIZE = 384                           # all-MiniLM-L6-v2 output dim
COLLECTION_NAME = "intchat_knowledge"
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "datasets" / "normalized_dataset.json"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"
BATCH_SIZE = 64
# ────────────────────────────────────────────────────────────────────────────


def doc_id_to_int(doc_id: str) -> int:
    """Stable int ID from a string doc ID (Qdrant needs int or uuid)."""
    return int(hashlib.md5(doc_id.encode()).hexdigest()[:15], 16)


def ingest(dataset_path: Path, reset: bool = False):
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connecting to Qdrant at {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL)

    # Collection setup
    existing = [c.name for c in client.get_collections().collections]
    if reset and COLLECTION_NAME in existing:
        print(f"Dropping existing collection '{COLLECTION_NAME}'")
        client.delete_collection(collection_name=COLLECTION_NAME)

    if COLLECTION_NAME not in existing or reset:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance="Cosine"),
        )
        print(f"Created collection '{COLLECTION_NAME}'")

    # Load normalized docs
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    if not isinstance(docs, list):
        raise ValueError(f"Expected a JSON array, got {type(docs).__name__}")

    print(f"Embedding and ingesting {len(docs)} documents...")

    points = []
    for doc in docs:
        text = doc.get("text", "")
        if not text:
            continue

        vector = model.encode(text).tolist()
        point = PointStruct(
            id=doc_id_to_int(doc["id"]),
            vector=vector,
            payload={
                "doc_id": doc["id"],
                "text": text,
                "category": doc.get("category", ""),
                "topic": doc.get("topic", ""),
                "type": doc.get("type", ""),
                "source": doc.get("source", ""),
                "metadata": doc.get("metadata", {}),
            },
        )
        points.append(point)

    # Batch upsert
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Upserting"):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    count = client.get_collection(COLLECTION_NAME).points_count
    print(f"\n✅ Done! Collection '{COLLECTION_NAME}' now has {count} points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest normalized dataset into Qdrant")
    parser.add_argument("--file", type=Path, default=DEFAULT_DATASET, help="Path to normalized_dataset.json")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate collection before ingesting")
    args = parser.parse_args()
    ingest(args.file, args.reset)

