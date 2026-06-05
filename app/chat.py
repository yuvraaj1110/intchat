"""CLI REPL for the international-student RAG chatbot.

Startup validates the API key and that a populated Chroma collection exists,
then loops: read question → validate → answer → print.
"""

import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app import config
from app.chain import RAGChain


def validate_query(raw: str) -> str | None:
    """Return a cleaned query, or None if it is empty."""
    cleaned = raw.strip()
    if not cleaned:
        return None
    return cleaned[: config.MAX_QUERY_LEN]


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
            print("\nAssistant: ", end="", flush=True)
            for piece in rag.answer_stream(query):
                print(piece, end="", flush=True)
            print()
        except Exception as exc:  # noqa: BLE001 - surface a friendly message
            print(f"\nSorry, something went wrong: {exc}\nPlease try again in a moment.")
            continue


if __name__ == "__main__":
    main()
