"""Fetch raw HTML from a URL.

Simple, single-responsibility module: download a web page and return
the raw HTML string. Error handling returns None so the orchestrator
can skip failed sources without crashing.
"""

import logging

import requests

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def fetch(url: str, timeout: int = 30) -> str | None:
    """Download *url* and return raw HTML, or None on any failure."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None
