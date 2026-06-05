# tests/test_prompts.py
from app import prompts


def test_system_prompt_has_guardrails():
    text = prompts.SYSTEM_PROMPT
    assert "ONLY" in text
    assert "DSO" in text
    assert "not legal advice" in text


def test_build_prompt_includes_context_and_question():
    msgs = prompts.build_prompt(
        context="OPT allows F-1 students to work.",
        chat_history="",
        question="What is OPT?",
    )
    rendered = "\n".join(m.content for m in msgs)
    assert "OPT allows F-1 students to work." in rendered
    assert "What is OPT?" in rendered
