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


@pytest.mark.slow
def test_reset_is_idempotent(tmp_path, sample_docs):
    """Re-running build_store with reset=True must not accumulate documents."""
    d = str(tmp_path)
    first = ingest.build_store(sample_docs, persist_dir=d, reset=True)
    count_first = first._collection.count()
    second = ingest.build_store(sample_docs, persist_dir=d, reset=True)
    count_second = second._collection.count()
    assert count_first == count_second
    # 6 sample docs - 1 deduped = 5 docs, none long enough to chunk at 800
    assert count_second == 5
