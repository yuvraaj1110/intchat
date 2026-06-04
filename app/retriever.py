"""Hybrid retriever: semantic search merged with a metadata/keyword filter.

Pure semantic search can miss exact legal terms (I-765, SEVP, EAD). This
retriever runs a semantic search and, when the query contains known terms,
also runs a keyword search over document text, then merges the two rankings.
"""

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app import config


def extract_known_terms(query: str) -> list[str]:
    """Return known immigration terms present in the query (case-insensitive)."""
    upper = query.upper()
    return [t for t in config.KNOWN_TERMS if t.upper() in upper]


class HybridRetriever(BaseRetriever):
    """Merge semantic hits with keyword hits via Reciprocal Rank Fusion."""

    store: Any = Field(...)
    semantic_k: int = Field(default=config.SEMANTIC_K)
    top_k: int = Field(default=config.TOP_K)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        semantic = self.store.similarity_search(query, k=self.semantic_k)

        keyword: list[Document] = []
        terms = extract_known_terms(query)
        if terms:
            # Pull a wide semantic net, then keep docs whose text contains a term.
            candidates = self.store.similarity_search(query, k=self.semantic_k * 2)
            keyword = [
                d for d in candidates
                if any(t.upper() in d.page_content.upper() for t in terms)
            ]

        return self._fuse(semantic, keyword)[: self.top_k]

    def _fuse(self, *rankings: list[Document], k: int = 60) -> list[Document]:
        """Reciprocal Rank Fusion across multiple ranked lists."""
        scores: dict[str, float] = {}
        by_id: dict[str, Document] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                key = doc.metadata.get("doc_id", doc.page_content[:50])
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                by_id[key] = doc
        ordered = sorted(scores, key=scores.get, reverse=True)
        return [by_id[key] for key in ordered]


def build_retriever(store) -> HybridRetriever:
    """Construct a HybridRetriever over an existing Chroma store."""
    return HybridRetriever(store=store)
