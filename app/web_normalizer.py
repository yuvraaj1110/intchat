"""Convert fetched web/PDF text into the normalized document schema.

Every source — hand-written JSON, scraped HTML, extracted PDF — must
converge on the same flat doc format so the downstream pipeline (dedup,
chunking, retrieval) is source-agnostic.  This module handles the
web/PDF side: split long text into sections, stamp provenance metadata.
"""

import re
from datetime import date, timezone, datetime

from app import config


def _slugify(text: str) -> str:
    """Turn a human label into a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _derive_source_key(name: str) -> str:
    """Derive a short source key like 'uscis' or 'purdue_iss' from the name."""
    first_part = name.split("—")[0].split("–")[0].strip()
    return _slugify(first_part)


def _split_sections(text: str, target_size: int = config.CHUNK_SIZE) -> list[str]:
    """Split *text* into sections roughly *target_size* chars each.

    Strategy: split on double-newlines (paragraph boundaries) first.
    Merge small paragraphs together until hitting the target.  If a
    single paragraph exceeds the target, keep it whole — the downstream
    chunker will split it further.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    sections: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current and current_len + len(para) + 2 > target_size:
            sections.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para) + 2  # +2 for the join separator

    if current:
        sections.append("\n\n".join(current))

    return sections


def normalize_web_source(
    text: str,
    source_entry: dict,
) -> list[dict]:
    """Convert cleaned text + source metadata into normalized docs.

    Parameters
    ----------
    text : str
        The cleaned article text (from parse_html or fetch_pdf).
    source_entry : dict
        One entry from ``sources.yaml`` with keys: name, type, url/path, category.

    Returns
    -------
    list[dict]
        Flat docs matching the existing normalized schema, plus provenance
        fields (source_url, source_name, fetched_at) in metadata.
    """
    name = source_entry["name"]
    category = source_entry.get("category", "General")
    source_url = source_entry.get("url", "")
    source_key = _derive_source_key(name)
    topic_slug = _slugify(name)
    fetched_at = date.today().isoformat()

    sections = _split_sections(text)
    docs = []

    for idx, section in enumerate(sections):
        doc = {
            "id": f"{source_key}__{topic_slug}__{idx}",
            "text": section,
            "category": category,
            "topic": name,
            "type": "paragraph",
            "source": source_key,
            "metadata": {
                "source_url": source_url,
                "source_name": name,
                "fetched_at": fetched_at,
            },
        }
        docs.append(doc)

    return docs
