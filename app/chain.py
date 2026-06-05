"""Assemble retriever + prompt + LLM + memory into the RAG chain.

A lightweight `ConversationWindow` holds the last N exchanges (bounding the
context window). `RAGChain.answer_stream` retrieves context, renders the prompt,
and streams the LLM response token-by-token for responsive CLI output;
`RAGChain.answer` is a convenience that collects the stream into a full string.
Memory is in-process; swap for Redis/Postgres in production.
"""

from collections import deque

from app import config, prompts
from app.llm import build_llm
from app.retriever import build_retriever


def format_context(docs) -> str:
    """Render retrieved docs as a context block with topic + provenance labels."""
    blocks = []
    for d in docs:
        topic = d.metadata.get("topic", "")
        source_url = d.metadata.get("source_url", "")
        fetched_at = d.metadata.get("fetched_at", "")
        # Build label: [Topic | source_url | date] or [Topic] for hand-written
        if source_url:
            label = f"[{topic} | {source_url} | {fetched_at}] "
        elif topic:
            label = f"[{topic}] "
        else:
            label = ""
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
        self.llm = llm or build_llm(streaming=True)
        self.memory = ConversationWindow()

    def answer_stream(self, question: str):
        """Stream the answer token-by-token, yielding text chunks.

        The full answer is accumulated and stored in memory once the stream
        completes, so conversation history stays intact.
        """
        docs = self.retriever.invoke(question)
        context = format_context(docs)
        messages = prompts.build_prompt(
            context=context,
            chat_history=self.memory.render(),
            question=question,
        )
        collected = []
        for chunk in self.llm.stream(messages):
            piece = chunk.content if hasattr(chunk, "content") else str(chunk)
            if piece:
                collected.append(piece)
                yield piece
        self.memory.add(question, "".join(collected))

    def answer(self, question: str) -> str:
        """Return the full answer as a single string (collects the stream)."""
        return "".join(self.answer_stream(question))
