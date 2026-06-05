# tests/test_llm.py
import pytest

from app import llm


def test_build_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.build_llm()


def test_select_model_returns_first_available(monkeypatch):
    # Groq offers the 2nd and 3rd preferences but not the 1st
    monkeypatch.setattr(
        llm, "list_available_models",
        lambda: {"llama-3.1-8b-instant", "llama3-70b-8192"},
    )
    chosen = llm.select_model(
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192"]
    )
    assert chosen == "llama-3.1-8b-instant"


def test_select_model_raises_when_none_available(monkeypatch):
    monkeypatch.setattr(llm, "list_available_models", lambda: {"some-other-model"})
    with pytest.raises(RuntimeError, match="None of the preferred Groq models"):
        llm.select_model(["llama-3.3-70b-versatile"])


def test_build_llm_uses_selected_model(monkeypatch):
    monkeypatch.setattr(llm.config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm, "select_model", lambda prefs: "llama-3.3-70b-versatile")
    model = llm.build_llm()
    # langchain-groq exposes the configured model as `.model_name`
    assert model.model_name == "llama-3.3-70b-versatile"
    assert model.max_retries == llm.config.LLM_MAX_RETRIES
