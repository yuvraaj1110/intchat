# tests/test_fetch_pdf.py
import tempfile
from pathlib import Path

from pypdf import PdfWriter

from app import fetch_pdf


def _make_pdf(text: str) -> Path:
    """Create a minimal single-page PDF with the given text."""
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # pypdf blank pages have no text, so we test the extraction path
    # by verifying our function handles it gracefully
    path = Path(tempfile.mktemp(suffix=".pdf"))
    with open(path, "wb") as f:
        writer.write(f)
    return path


def test_extract_missing_file_returns_none():
    result = fetch_pdf.extract("/nonexistent/path/fake.pdf")
    assert result is None


def test_extract_blank_pdf_returns_none():
    path = _make_pdf("")
    # A blank page has no extractable text
    result = fetch_pdf.extract(path)
    assert result is None
    path.unlink()


def test_extract_accepts_path_object():
    result = fetch_pdf.extract(Path("/nonexistent/fake.pdf"))
    assert result is None
