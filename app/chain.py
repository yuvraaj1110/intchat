"""Assemble retriever + prompt + LLM + memory into a single answer() call.

A lightweight `ConversationWindow` holds the last N exchanges (bounding the
context window). `RAGChain.answer` retrieves context, renders the prompt, and
invokes the LLM. Memory is in-process; swap for Redis/Postgres in production.
"""

from collections import deque

from app import config, prompts
from app.llm import build_llm
from app.retriever import build_retriever


def format_context(docs) -> str:
    """Render retrieved docs as a context block with topic labels."""
    blocks = []
    for d in docs:
        topic = d.metadata.get("topic", "")
        label = f"[{topic}] " if topic else ""
        blocks.append(f"{label}{d.page_content}")
    return "\n\n".join(blocks)


class ConversationWindow:
    """Keep the most recent `max_pairs` (question, answer) exchanges."""

    def __init__(self, max_pairs: int = config.MEMORY_WINDOW):
        self._pairs: deque = deque(maxlen=max_pairs)

    def add(self, question: str, answer: str) -> None:
        self._pairs.append((question, answer))

    def render(self) -> str:
        return "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in self._pairs
        )


class RAGChain:
    """End-to-end retrieval-augmented chat."""

    def __init__(self, store, llm=None):
        self.retriever = build_retriever(store)
        self.llm = llm or build_llm(streaming=False)
        self.memory = ConversationWindow()

    def answer(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = format_context(docs)
        messages = prompts.build_prompt(
            context=context,
            chat_history=self.memory.render(),
            question=question,
        )
        response = self.llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        self.memory.add(question, text)
        return text
