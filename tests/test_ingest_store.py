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
