"""
Integration tests for master orchestration pipeline in main.py
"""

from unittest.mock import MagicMock, patch
import pytest

import main
from src.data_ingestion.fpl_api import BootstrapData


@pytest.fixture
def mock_pipeline_data():
    elements = []
    pid = 1

    # 4 GKs
    for i in range(1, 5):
        elements.append({
            "id": pid,
            "web_name": f"GK_{i}",
            "element_type": 1,
            "team": i,
            "now_cost": 45 + i,
            "ep_next": str(4.0 + i * 0.2),
            "form": "4.0",
        })
        pid += 1

    # 10 DEFs
    for i in range(1, 11):
        elements.append({
            "id": pid,
            "web_name": f"DEF_{i}",
            "element_type": 2,
            "team": (i % 6) + 1,
            "now_cost": 45 + (i % 4) * 5,
            "ep_next": str(4.5 + (i % 3) * 0.5),
            "form": "4.5",
        })
        pid += 1

    # 10 MIDs
    for i in range(1, 11):
        elements.append({
            "id": pid,
            "web_name": f"MID_{i}",
            "element_type": 3,
            "team": (i % 6) + 1,
            "now_cost": 55 + (i % 5) * 10,
            "ep_next": str(5.0 + (i % 4) * 0.8),
            "form": "5.0",
        })
        pid += 1

    # 6 FWDs
    for i in range(1, 7):
        elements.append({
            "id": pid,
            "web_name": f"FWD_{i}",
            "element_type": 4,
            "team": (i % 5) + 1,
            "now_cost": 60 + i * 10,
            "ep_next": str(5.5 + i * 0.5),
            "form": "5.5",
        })
        pid += 1

    data_dict = {
        "events": [],
        "teams": [],
        "elements": elements,
        "element_types": [],
    }
    return BootstrapData.from_dict(data_dict)


def test_main_dry_run_integration(mock_pipeline_data):
    with patch("main.FPLDataFetcher") as mock_fetcher_cls, \
         patch("main.PerplexityNewsClient") as mock_news_cls, \
         patch("main.ExpectedPointsGenerator") as mock_xp_cls:

        # Configure mock fetcher
        mock_fetcher = MagicMock()
        mock_fetcher.get_bootstrap_data.return_value = mock_pipeline_data
        mock_fetcher_cls.return_value = mock_fetcher

        # Configure mock news client
        mock_news = MagicMock()
        mock_news.fetch_injury_and_rotation_news.return_value = {
            "summary_text": "All key players fit.",
        }
        mock_news_cls.return_value = mock_news

        # Configure mock expected points generator
        mock_xp = MagicMock()
        mock_xp.generate_projections.return_value = {
            p.id: float(p.raw_data.get("ep_next", 4.0)) for p in mock_pipeline_data.elements
        }
        mock_xp_cls.return_value = mock_xp

        # Execute main with dry-run flag
        exit_code = main.main(["--dry-run", "--budget", "100.0"])

        assert exit_code == 0
        mock_fetcher.get_bootstrap_data.assert_called_once()
        mock_news.fetch_injury_and_rotation_news.assert_called_once()
        mock_xp.generate_projections.assert_called_once()
