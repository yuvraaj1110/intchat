"""Orchestrator: fetch all sources → parse → normalize → merge → ingest.

Entry point: python3 -m app.build_kb [--reset]

This is the only module that knows about I/O, ordering, and coordination.
The fetch/parse/normalize modules it calls are pure functions that are
easy to test in isolation.
"""

import argparse
import json
import logging
import sys

import yaml

from app import config
from app.fetch_html import fetch
from app.fetch_pdf import extract
from app.parse_html import extract_content
from app.web_normalizer import normalize_web_source
from app.ingest import build_store

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_sources() -> list[dict]:
    """Read sources.yaml and return the list of source entries."""
    if not config.SOURCES_YAML.exists():
        sys.exit(f"sources.yaml not found at {config.SOURCES_YAML}")
    with open(config.SOURCES_YAML, encoding="utf-8") as f:
        sources = yaml.safe_load(f)
    if not sources:
        sys.exit("sources.yaml is empty — add at least one source.")
    return sources


def fetch_and_parse(entry: dict) -> str | None:
    """Fetch and parse a single source entry. Returns clean text or None."""
    source_type = entry.get("type", "html")
    name = entry.get("name", "unknown")

    if source_type == "html":
        url = entry.get("url")
        if not url:
            logger.warning("Skipping '%s': no url field", name)
            return None
        raw_html = fetch(url)
        if raw_html is None:
            return None
        return extract_content(raw_html)

    elif source_type == "pdf":
        path = entry.get("path")
        if not path:
            logger.warning("Skipping '%s': no path field", name)
            return None
        full_path = config.BASE_DIR / path
        return extract(full_path)

    else:
        logger.warning("Skipping '%s': unknown type '%s'", name, source_type)
        return None


def load_hand_written_docs() -> list[dict]:
    """Load the existing hand-written normalized dataset."""
    if config.NORMALIZED_DATASET.exists():
        with open(config.NORMALIZED_DATASET, encoding="utf-8") as f:
            return json.load(f)
    return []


def build_knowledge_base(reset: bool = False) -> None:
    """Full pipeline: fetch → parse → normalize → merge → ingest."""
    sources = load_sources()
    logger.info("Loaded %d sources from sources.yaml", len(sources))

    web_docs: list[dict] = []
    skipped = 0

    for entry in sources:
        name = entry.get("name", "unknown")
        text = fetch_and_parse(entry)
        if text is None:
            logger.warning("⚠️  Skipped: %s", name)
            skipped += 1
            continue
        docs = normalize_web_source(text, entry)
        web_docs.extend(docs)
        logger.info("✓  %s → %d docs", name, len(docs))

    # Load and merge with hand-written docs
    hand_docs = load_hand_written_docs()
    all_docs = hand_docs + web_docs

    # Write merged dataset
    with open(config.NORMALIZED_ALL, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)
    logger.info(
        "Merged: %d hand-written + %d web = %d total docs",
        len(hand_docs), len(web_docs), len(all_docs),
    )

    # Build the Chroma store
    store = build_store(all_docs, reset=reset)
    count = store._collection.count()

    print(f"\n{'='*60}")
    print(f"✅ Knowledge base rebuilt successfully!")
    print(f"   Sources processed: {len(sources) - skipped}")
    print(f"   Sources skipped:   {skipped}")
    print(f"   Total documents:   {len(all_docs)}")
    print(f"   Chunks in Chroma:  {count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build the knowledge base from all sources"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe and rebuild the Chroma collection from scratch",
    )
    args = parser.parse_args()
    build_knowledge_base(reset=args.reset)


if __name__ == "__main__":
    main()
