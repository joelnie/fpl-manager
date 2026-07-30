"""
Unit tests for FPLDataFetcher in src/data_ingestion/fpl_api.py
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.data_ingestion.fpl_api import (
    FPLDataFetcher,
    FPLAPIError,
    BootstrapData,
    Fixture,
    PlayerSummary,
    MyTeam,
)


@pytest.fixture
def fetcher():
    return FPLDataFetcher(base_url="https://fantasy.premierleague.com/api", timeout=5)


@pytest.fixture
def mock_bootstrap_response():
    return {
        "elements": [
            {
                "id": 1,
                "web_name": "Saka",
                "first_name": "Bukayo",
                "second_name": "Saka",
                "team": 1,
                "element_type": 3,
                "now_cost": 100,
                "selected_by_percent": "60.5",
                "form": "7.5",
                "points_per_game": "6.2",
                "total_points": 180,
            }
        ],
        "teams": [
            {
                "id": 1,
                "name": "Arsenal",
                "short_name": "ARS",
                "code": 3,
                "strength": 5,
            }
        ],
        "events": [
            {
                "id": 1,
                "name": "Gameweek 1",
                "deadline_time": "2024-08-16T17:30:00Z",
                "is_previous": False,
                "is_current": True,
                "is_next": False,
                "finished": True,
            }
        ],
        "element_types": [
            {"id": 1, "singular_name": "Goalkeeper"},
            {"id": 2, "singular_name": "Defender"},
            {"id": 3, "singular_name": "Midfielder"},
            {"id": 4, "singular_name": "Forward"},
        ],
    }


@pytest.fixture
def mock_fixtures_response():
    return [
        {
            "id": 1,
            "code": 2444470,
            "team_h": 1,
            "team_a": 2,
            "team_h_score": 2,
            "team_a_score": 0,
            "event": 1,
            "finished": True,
            "kickoff_time": "2024-08-17T14:00:00Z",
        }
    ]


@pytest.fixture
def mock_player_summary_response():
    return {
        "fixtures": [{"id": 10, "event": 2, "team_h": 1, "team_a": 3}],
        "history": [
            {
                "element": 1,
                "fixture": 1,
                "total_points": 10,
                "was_home": True,
                "round": 1,
            }
        ],
        "history_past": [],
    }


@pytest.fixture
def mock_my_team_response():
    return {
        "picks": [
            {
                "element": 1,
                "position": 1,
                "selling_price": 100,
                "purchase_price": 100,
                "multiplier": 1,
                "is_captain": False,
                "is_vice_captain": True,
            }
        ],
        "chips": [],
        "transfers": {"made": 0, "bank": 10},
    }


def test_get_bootstrap_data_success(fetcher, mock_bootstrap_response):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_res = MagicMock()
        mock_res.ok = True
        mock_res.json.return_value = mock_bootstrap_response
        mock_get.return_value = mock_res

        data = fetcher.get_bootstrap_data()

        assert isinstance(data, BootstrapData)
        assert len(data.elements) == 1
        assert data.elements[0].web_name == "Saka"
        assert len(data.teams) == 1
        assert data.teams[0].short_name == "ARS"
        assert len(data.events) == 1
        assert data.events[0].id == 1

        # DataFrame conversions
        df_players = data.to_player_df()
        assert not df_players.empty
        assert df_players.iloc[0]["web_name"] == "Saka"

        df_teams = data.to_team_df()
        assert not df_teams.empty
        assert df_teams.iloc[0]["short_name"] == "ARS"


def test_get_fixtures_success(fetcher, mock_fixtures_response):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_res = MagicMock()
        mock_res.ok = True
        mock_res.json.return_value = mock_fixtures_response
        mock_get.return_value = mock_res

        fixtures = fetcher.get_fixtures()

        assert isinstance(fixtures, list)
        assert len(fixtures) == 1
        assert isinstance(fixtures[0], Fixture)
        assert fixtures[0].team_h == 1
        assert fixtures[0].team_h_score == 2
        mock_get.assert_called_with("https://fantasy.premierleague.com/api/fixtures/", timeout=5, cookies=None, headers=None)


def test_get_fixtures_with_event(fetcher, mock_fixtures_response):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_res = MagicMock()
        mock_res.ok = True
        mock_res.json.return_value = mock_fixtures_response
        mock_get.return_value = mock_res

        fixtures = fetcher.get_fixtures(event=1)

        assert isinstance(fixtures, list)
        assert len(fixtures) == 1
        mock_get.assert_called_with("https://fantasy.premierleague.com/api/fixtures/?event=1", timeout=5, cookies=None, headers=None)


def test_get_player_summary_success(fetcher, mock_player_summary_response):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_res = MagicMock()
        mock_res.ok = True
        mock_res.json.return_value = mock_player_summary_response
        mock_get.return_value = mock_res

        summary = fetcher.get_player_summary(1)

        assert isinstance(summary, PlayerSummary)
        assert summary.player_id == 1
        assert len(summary.history) == 1
        assert summary.history[0]["total_points"] == 10

        history_df = summary.to_history_df()
        assert not history_df.empty
        assert history_df.iloc[0]["total_points"] == 10


def test_get_my_team_authenticated_success(fetcher, mock_my_team_response):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_res = MagicMock()
        mock_res.ok = True
        mock_res.json.return_value = mock_my_team_response
        mock_get.return_value = mock_res

        cookies = {"pl_profile": "test_cookie"}
        my_team = fetcher.get_my_team(manager_id=12345, session_cookie=cookies)

        assert isinstance(my_team, MyTeam)
        assert my_team.user_id == 12345
        assert len(my_team.picks) == 1
        assert my_team.picks[0].element == 1

        picks_df = my_team.to_picks_df()
        assert not picks_df.empty


def test_get_my_team_unauthenticated_error(fetcher):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_res = MagicMock()
        mock_res.ok = False
        mock_res.status_code = 401
        mock_res.response = mock_res
        mock_get.return_value = mock_res

        with pytest.raises(FPLAPIError) as exc_info:
            fetcher.get_my_team(user_id=12345)

        assert exc_info.value.status_code == 401
        assert "Authentication required" in str(exc_info.value)


def test_http_500_error_handling(fetcher):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_res = MagicMock()
        mock_res.ok = False
        mock_res.status_code = 500
        mock_get.return_value = mock_res

        with pytest.raises(FPLAPIError) as exc_info:
            fetcher.get_bootstrap_data()

        assert exc_info.value.status_code == 500


def test_timeout_error_handling(fetcher):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        with pytest.raises(FPLAPIError) as exc_info:
            fetcher.get_bootstrap_data()

        assert "Request timeout" in str(exc_info.value)


def test_invalid_json_handling(fetcher):
    with patch.object(fetcher.session, "get") as mock_get:
        mock_res = MagicMock()
        mock_res.ok = True
        mock_res.json.side_effect = ValueError("No JSON could be decoded")
        mock_get.return_value = mock_res

        with pytest.raises(FPLAPIError) as exc_info:
            fetcher.get_bootstrap_data()

        assert "Failed to parse JSON" in str(exc_info.value)
