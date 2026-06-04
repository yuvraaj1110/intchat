# tests/test_retriever.py
import pytest

from app import ingest, retriever


@pytest.fixture
def built_store(tmp_path, sample_docs):
    return ingest.build_store(sample_docs, persist_dir=str(tmp_path), reset=True)


def test_extract_known_terms_finds_form_numbers():
    terms = retriever.extract_known_terms("Do I need form I-765 for OPT?")
    assert "I-765" in terms
    assert "OPT" in terms


def test_extract_known_terms_empty_when_none_present():
    assert retriever.extract_known_terms("how do I make friends on campus?") == []


@pytest.mark.slow
def test_retrieve_semantic_paraphrase(built_store):
    r = retriever.build_retriever(built_store)
    docs = r.invoke("can I work in my field after graduation?")
    assert any("OPT" in d.page_content or d.metadata.get("category") in {"OPT", "Practical Training and Employment"}
               for d in docs)


@pytest.mark.slow
def test_retrieve_exact_term_surfaces_i765_doc(built_store):
    r = retriever.build_retriever(built_store)
    docs = r.invoke("which form is I-765 used for?")
    assert any("I-765" in d.page_content for d in docs)
