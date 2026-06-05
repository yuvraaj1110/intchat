"""Strip boilerplate from raw HTML and return clean article text.

Uses trafilatura, which is purpose-built for extracting the main content
from web pages — it removes navigation, footers, sidebars, cookie
banners, and other chrome.  This is the primary quality gate: clean text
in → good retrieval; dirty text → polluted answers.
"""

import logging

import trafilatura

logger = logging.getLogger(__name__)


def extract_content(raw_html: str) -> str | None:
    """Return the main article text from *raw_html*, or None.

    Returns None when trafilatura finds no meaningful content (e.g. a
    page that is entirely JavaScript-rendered or just a redirect).
    """
    if not raw_html:
        return None
    text = trafilatura.extract(
        raw_html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    if not text or not text.strip():
        logger.warning("No meaningful content extracted from HTML")
        return None
    return text.strip()
