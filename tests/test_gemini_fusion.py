"""
Unit tests for ExpectedPointsGenerator in src/intelligence/gemini_fusion.py
"""

from unittest.mock import MagicMock, patch
import pytest

from src.intelligence.gemini_fusion import ExpectedPointsGenerator


@pytest.fixture
def mock_bootstrap_data():
    return {
        "elements": [
            {
                "id": 1,
                "web_name": "Saka",
                "form": "7.5",
                "ep_next": "8.0",
                "ict_index": "85.2",
                "points_per_game": "6.5",
            },
            {
                "id": 2,
                "web_name": "Haaland",
                "form": "9.0",
                "ep_next": "9.5",
                "ict_index": "95.0",
                "points_per_game": "8.2",
            },
            {
                "id": 3,
                "web_name": "Palmer",
                "form": "6.0",
                "ep_next": None,
                "ict_index": "70.0",
                "points_per_game": "5.5",
            },
        ]
    }


def test_generate_projections_api_success(mock_bootstrap_data):
    generator = ExpectedPointsGenerator(api_key="test_key")
    generator.client = MagicMock()

    mock_response = MagicMock()
    mock_response.text = (
        '{"projections": ['
        '{"player_id": 1, "expected_points": 8.5, "confidence_score": 0.9, "rationale": "High form"},'
        '{"player_id": 2, "expected_points": 10.0, "confidence_score": 0.95, "rationale": "Captain material"},'
        '{"player_id": 3, "expected_points": 6.2, "confidence_score": 0.75, "rationale": "Decent fixture"}'
        ']}'
    )
    generator.client.models.generate_content.return_value = mock_response

    projections = generator.generate_projections(mock_bootstrap_data, news_context="No major injuries")

    assert isinstance(projections, dict)
    assert len(projections) == 3
    assert projections[1] == 8.5
    assert projections[2] == 10.0
    assert projections[3] == 6.2

    # Check model generate_content call
    generator.client.models.generate_content.assert_called_once()


def test_fallback_projections_without_api_key(mock_bootstrap_data):
    generator = ExpectedPointsGenerator(api_key=None)
    generator.client = None

    projections = generator.generate_projections(mock_bootstrap_data)

    assert isinstance(projections, dict)
    assert len(projections) == 3
    assert projections[1] == 8.0  # from ep_next
    assert projections[2] == 9.5  # from ep_next
    assert projections[3] == 6.0  # from form (ep_next is None)


def test_api_error_fallback(mock_bootstrap_data):
    generator = ExpectedPointsGenerator(api_key="test_key")
    generator.client = MagicMock()
    generator.client.models.generate_content.side_effect = RuntimeError("API Rate limit or network error")

    projections = generator.generate_projections(mock_bootstrap_data)

    assert isinstance(projections, dict)
    assert len(projections) == 3
    assert projections[1] == 8.0
    assert projections[2] == 9.5
    assert projections[3] == 6.0
