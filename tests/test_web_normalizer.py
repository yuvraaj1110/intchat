# tests/test_web_normalizer.py
from app import web_normalizer


_ENTRY = {
    "name": "USCIS — OPT for F-1 Students",
    "type": "html",
    "url": "https://www.uscis.gov/opt",
    "category": "OPT",
}


def test_normalize_produces_correct_schema():
    text = "F-1 students may apply for OPT after completing their program."
    docs = web_normalizer.normalize_web_source(text, _ENTRY)
    assert len(docs) >= 1
    doc = docs[0]
    # Required schema fields
    assert "id" in doc
    assert "text" in doc
    assert doc["category"] == "OPT"
    assert doc["topic"] == "USCIS — OPT for F-1 Students"
    assert doc["type"] == "paragraph"
    assert doc["source"] == "uscis"
    # Provenance metadata
    assert doc["metadata"]["source_url"] == "https://www.uscis.gov/opt"
    assert doc["metadata"]["source_name"] == "USCIS — OPT for F-1 Students"
    assert doc["metadata"]["fetched_at"]  # ISO date string


def test_normalize_splits_long_text():
    # Create text with multiple paragraphs exceeding CHUNK_SIZE
    para = "This is a paragraph about immigration rules. " * 30  # ~1350 chars
    text = f"{para}\n\n{para}"  # Two big paragraphs
    docs = web_normalizer.normalize_web_source(text, _ENTRY)
    assert len(docs) >= 2


def test_normalize_short_text_single_doc():
    text = "Short answer about OPT."
    docs = web_normalizer.normalize_web_source(text, _ENTRY)
    assert len(docs) == 1
    assert docs[0]["text"] == "Short answer about OPT."


def test_normalize_ids_are_unique():
    text = "Para one about OPT.\n\nPara two about CPT.\n\nPara three about STEM."
    docs = web_normalizer.normalize_web_source(text, _ENTRY)
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids))


def test_derive_source_key():
    assert web_normalizer._derive_source_key("USCIS — OPT for F-1") == "uscis"
    assert web_normalizer._derive_source_key("Purdue ISS — CPT") == "purdue_iss"
    assert web_normalizer._derive_source_key("Study in the States — Travel") == "study_in_the_states"
