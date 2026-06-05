# tests/test_build_kb.py
from unittest.mock import patch, Mock

from app import build_kb


def test_load_sources_reads_yaml(tmp_path):
    yaml_content = """
- name: "Test Source"
  type: html
  url: "https://example.com"
  category: "TEST"
"""
    yaml_file = tmp_path / "sources.yaml"
    yaml_file.write_text(yaml_content)
    with patch("app.build_kb.config") as mock_config:
        mock_config.SOURCES_YAML = yaml_file
        sources = build_kb.load_sources()
    assert len(sources) == 1
    assert sources[0]["name"] == "Test Source"


def test_fetch_and_parse_html_success():
    entry = {"name": "Test", "type": "html", "url": "https://example.com"}
    with patch("app.build_kb.fetch", return_value="<html><body><p>Hello world</p></body></html>"):
        with patch("app.build_kb.extract_content", return_value="Hello world"):
            result = build_kb.fetch_and_parse(entry)
    assert result == "Hello world"


def test_fetch_and_parse_html_failure():
    entry = {"name": "Test", "type": "html", "url": "https://example.com/broken"}
    with patch("app.build_kb.fetch", return_value=None):
        result = build_kb.fetch_and_parse(entry)
    assert result is None


def test_fetch_and_parse_pdf():
    entry = {"name": "Test PDF", "type": "pdf", "path": "datasets/pdfs/test.pdf"}
    with patch("app.build_kb.extract", return_value="PDF text content"):
        with patch("app.build_kb.config") as mock_config:
            from pathlib import Path
            mock_config.BASE_DIR = Path("/fake")
            result = build_kb.fetch_and_parse(entry)
    assert result == "PDF text content"


def test_fetch_and_parse_unknown_type():
    entry = {"name": "Mystery", "type": "ftp", "url": "ftp://example.com"}
    result = build_kb.fetch_and_parse(entry)
    assert result is None
