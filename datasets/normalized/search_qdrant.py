"""
Interactive semantic search over the normalized knowledge base in Qdrant.

Usage:
    python search_qdrant.py                   # interactive REPL
    python search_qdrant.py "how does OPT work"  # single query
"""

import sys
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "intchat_knowledge"
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "all-MiniLM-L6-v2"

client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(MODEL_NAME)


def search(query: str, limit: int = 5):
    query_vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    points = results.points
    print(f"\nTop {len(points)} results for: \"{query}\"\n{'─'*60}")

    for i, hit in enumerate(points, start=1):
        p = hit.payload or {}
        source = p.get("source", "?")
        doc_type = p.get("type", "?")
        category = p.get("category", "")
        topic = p.get("topic", "")
        text = p.get("text", "[NO TEXT]")

        print(f"\n[{i}] score={hit.score:.4f}  |  {source} / {doc_type}")
        if category:
            print(f"    Category: {category}")
        if topic and topic != category:
            print(f"    Topic:    {topic}")
        # Truncate long text for readability
        preview = text[:300] + ("..." if len(text) > 300 else "")
        print(f"    {preview}")

    return points


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query from command line
        search(" ".join(sys.argv[1:]))
    else:
        # Interactive REPL
        print("Intchat Knowledge Search (type 'exit' to quit)")
        while True:
            try:
                q = input("\nAsk a question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() == "exit":
                break
            if q:
                search(q)

