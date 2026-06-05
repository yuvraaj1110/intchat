# tests/test_chat.py
from app import chat


def test_validate_query_rejects_empty():
    assert chat.validate_query("") is None
    assert chat.validate_query("   ") is None


def test_validate_query_truncates_long_input():
    long = "a" * 1000
    assert len(chat.validate_query(long)) == 500


def test_validate_query_passes_normal_input():
    assert chat.validate_query("What is OPT?") == "What is OPT?"


def test_load_store_errors_when_missing(tmp_path, monkeypatch):
    # Point at an empty dir → no collection → clear error
    monkeypatch.setattr(chat.config, "CHROMA_DIR", tmp_path / "nope")
    import pytest
    with pytest.raises(SystemExit):
        chat.load_store()
