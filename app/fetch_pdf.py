"""Extract text from a local PDF file.

Single-responsibility module: read a PDF and return its full text.
Returns None on failure so the orchestrator can skip bad files.
"""

import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract(path: str | Path) -> str | None:
    """Return the concatenated text of all pages in *path*, or None."""
    path = Path(path)
    if not path.exists():
        logger.warning("PDF not found: %s", path)
        return None
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p for p in pages if p.strip())
        return text if text.strip() else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read PDF %s: %s", path, exc)
        return None
