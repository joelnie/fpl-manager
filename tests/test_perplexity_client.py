"""
Unit tests for PerplexityNewsClient in src/intelligence/perplexity_client.py
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.intelligence.perplexity_client import PerplexityNewsClient


@pytest.fixture
def client():
    return PerplexityNewsClient(api_key="test_perplexity_key", timeout=5)


def test_fetch_news_success(client):
    players = ["Saka", "Haaland"]
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Saka trained fully on Thursday. Haaland expected to start despite knock."
                }
            }
        ]
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value = mock_response

        res = client.fetch_injury_and_rotation_news(players)

        assert isinstance(res, dict)
        assert "summary_text" in res
        assert "Saka trained fully" in res["summary_text"]
        assert "Saka" in res
        assert "Haaland" in res
        assert res["Saka"]["status"] == "Check News"

        # Verify POST payload and headers
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test_perplexity_key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["json"]["model"] == "sonar-pro"
        assert len(kwargs["json"]["messages"]) == 2


def test_fallback_without_api_key():
    client_no_key = PerplexityNewsClient(api_key=None)
    client_no_key.api_key = None

    with patch("requests.post") as mock_post:
        res = client_no_key.fetch_injury_and_rotation_news(["Saka", "Haaland"])

        # Should not trigger HTTP POST
        mock_post.assert_not_called()
        assert isinstance(res, dict)
        assert res["Saka"]["status"] == "Available"
        assert res["Haaland"]["risk_multiplier"] == 1.0


def test_fallback_on_http_error(client):
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.RequestException("API connection timeout")

        res = client.fetch_injury_and_rotation_news(["Saka", "Haaland"])

        assert isinstance(res, dict)
        assert res["Saka"]["status"] == "Available"
        assert res["Haaland"]["risk_multiplier"] == 1.0
        assert "fallback mode active" in res["summary_text"]
