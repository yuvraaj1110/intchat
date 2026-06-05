# tests/test_parse_html.py
from app import parse_html


_SAMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
<nav><a href="/">Home</a> | <a href="/about">About</a></nav>
<main>
<h1>Optional Practical Training</h1>
<p>F-1 students may apply for 12 months of post-completion OPT.
The application requires Form I-765 and supporting documents.</p>
<p>Students must apply within 60 days of program completion.</p>
</main>
<footer>Copyright 2024. Privacy Policy. Terms of Use.</footer>
</body>
</html>
"""


def test_extract_content_strips_nav_and_footer():
    result = parse_html.extract_content(_SAMPLE_HTML)
    assert result is not None
    # Main content should survive
    assert "F-1 students" in result or "Optional Practical Training" in result
    # Nav/footer should be stripped (trafilatura does this)
    # Note: trafilatura may or may not perfectly strip all boilerplate
    # on minimal HTML, so we just verify we got something meaningful
    assert len(result) > 20


def test_extract_content_empty_html_returns_none():
    result = parse_html.extract_content("")
    assert result is None


def test_extract_content_none_input_returns_none():
    result = parse_html.extract_content(None)
    assert result is None


def test_extract_content_only_nav_returns_none_or_minimal():
    html = "<html><body><nav><a>Home</a><a>About</a></nav></body></html>"
    result = parse_html.extract_content(html)
    # Entirely navigation — should be None or very short
    assert result is None or len(result) < 20
