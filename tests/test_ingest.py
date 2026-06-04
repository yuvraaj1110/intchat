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
