"""Groq chat model construction with self-healing model selection + retry.

Groq hosts third-party open models and retires them on its own schedule, so no
hardcoded model name stays valid forever. At startup we query Groq's live model
list and pick the first entry from `config.GROQ_MODEL_PREFERENCES` that Groq
actually serves. `ChatGroq` then retries transient errors (including HTTP 429
rate limits) up to `max_retries` times with exponential backoff.
"""

from groq import Groq
from langchain_groq import ChatGroq

from app import config


def list_available_models() -> set[str]:
    """Return the set of model ids Groq currently serves for this account."""
    client = Groq(api_key=config.GROQ_API_KEY)
    return {m.id for m in client.models.list().data}


def select_model(preferences: list[str]) -> str:
    """Pick the first preferred model Groq actually offers.

    Falls back through the preference list; raises if none are available.
    """
    available = list_available_models()
    for name in preferences:
        if name in available:
            return name
    raise RuntimeError(
        "None of the preferred Groq models are available. "
        f"Preferences: {preferences}. Available: {sorted(available)}. "
        "Update config.GROQ_MODEL_PREFERENCES — see "
        "https://console.groq.com/docs/models"
    )


def build_llm(streaming: bool = True) -> ChatGroq:
    """Construct the Groq chat model. Raises if the API key is missing."""
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys"
        )
    model_name = select_model(config.GROQ_MODEL_PREFERENCES)
    return ChatGroq(
        model=model_name,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_retries=config.LLM_MAX_RETRIES,
        streaming=streaming,
    )
