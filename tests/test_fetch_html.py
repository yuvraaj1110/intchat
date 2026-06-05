# tests/test_fetch_html.py
from unittest.mock import patch, Mock

from app import fetch_html


def test_fetch_success():
    mock_resp = Mock()
    mock_resp.text = "<html><body>Hello</body></html>"
    mock_resp.raise_for_status = Mock()
    with patch("app.fetch_html.requests.get", return_value=mock_resp) as m:
        result = fetch_html.fetch("https://example.com")
    assert result == "<html><body>Hello</body></html>"
    m.assert_called_once()


def test_fetch_404_returns_none():
    import requests
    mock_resp = Mock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
    with patch("app.fetch_html.requests.get", return_value=mock_resp):
        result = fetch_html.fetch("https://example.com/missing")
    assert result is None


def test_fetch_timeout_returns_none():
    import requests
    with patch("app.fetch_html.requests.get", side_effect=requests.Timeout("timeout")):
        result = fetch_html.fetch("https://example.com/slow")
    assert result is None


def test_fetch_sends_user_agent():
    mock_resp = Mock()
    mock_resp.text = "<html></html>"
    mock_resp.raise_for_status = Mock()
    with patch("app.fetch_html.requests.get", return_value=mock_resp) as m:
        fetch_html.fetch("https://example.com")
    call_kwargs = m.call_args[1]
    assert "User-Agent" in call_kwargs["headers"]
    assert "Mozilla" in call_kwargs["headers"]["User-Agent"]
