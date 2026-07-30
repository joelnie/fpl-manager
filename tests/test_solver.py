"""
Unit tests for FPLOptimizer ILP Solver in src/optimization/ilp_solver.py
"""

from typing import Any, Dict, List
import pytest
from src.optimization.ilp_solver import FPLOptimizer, OptimizationResult
from src.data_ingestion.fpl_api import Player


@pytest.fixture
def mock_player_dataset() -> List[Dict[str, Any]]:
    """Synthetic dataset of 28 players across positions, teams, costs, and expected points."""
    players = []
    pid = 1

    # 4 Goalkeepers (element_type = 1)
    gks = [
        {"team": 1, "cost": 45, "xP": 4.5, "web_name": "GK1"},
        {"team": 2, "cost": 50, "xP": 5.0, "web_name": "GK2"},
        {"team": 3, "cost": 40, "xP": 3.0, "web_name": "GK3"},
        {"team": 4, "cost": 55, "xP": 5.2, "web_name": "GK4"},
    ]
    for gk in gks:
        players.append({"id": pid, "element_type": 1, **gk})
        pid += 1

    # 10 Defenders (element_type = 2)
    defs = [
        {"team": 1, "cost": 60, "xP": 6.0, "web_name": "DEF1"},
        {"team": 1, "cost": 55, "xP": 5.5, "web_name": "DEF2"},
        {"team": 1, "cost": 50, "xP": 5.0, "web_name": "DEF3"},
        {"team": 2, "cost": 50, "xP": 4.8, "web_name": "DEF4"},
        {"team": 2, "cost": 45, "xP": 4.2, "web_name": "DEF5"},
        {"team": 3, "cost": 45, "xP": 4.0, "web_name": "DEF6"},
        {"team": 3, "cost": 40, "xP": 3.8, "web_name": "DEF7"},
        {"team": 4, "cost": 45, "xP": 4.5, "web_name": "DEF8"},
        {"team": 5, "cost": 50, "xP": 4.9, "web_name": "DEF9"},
        {"team": 6, "cost": 40, "xP": 3.5, "web_name": "DEF10"},
    ]
    for df in defs:
        players.append({"id": pid, "element_type": 2, **df})
        pid += 1

    # 10 Midfielders (element_type = 3)
    mids = [
        {"team": 1, "cost": 125, "xP": 9.0, "web_name": "MID1"},
        {"team": 2, "cost": 100, "xP": 7.5, "web_name": "MID2"},
        {"team": 2, "cost": 85, "xP": 6.8, "web_name": "MID3"},
        {"team": 3, "cost": 75, "xP": 6.0, "web_name": "MID4"},
        {"team": 3, "cost": 65, "xP": 5.2, "web_name": "MID5"},
        {"team": 4, "cost": 70, "xP": 5.5, "web_name": "MID6"},
        {"team": 4, "cost": 60, "xP": 4.8, "web_name": "MID7"},
        {"team": 5, "cost": 55, "xP": 4.5, "web_name": "MID8"},
        {"team": 5, "cost": 50, "xP": 4.0, "web_name": "MID9"},
        {"team": 6, "cost": 45, "xP": 3.8, "web_name": "MID10"},
    ]
    for md in mids:
        players.append({"id": pid, "element_type": 3, **md})
        pid += 1

    # 6 Forwards (element_type = 4)
    fwds = [
        {"team": 1, "cost": 140, "xP": 9.5, "web_name": "FWD1"},
        {"team": 2, "cost": 80, "xP": 6.5, "web_name": "FWD2"},
        {"team": 3, "cost": 75, "xP": 6.0, "web_name": "FWD3"},
        {"team": 4, "cost": 60, "xP": 5.0, "web_name": "FWD4"},
        {"team": 5, "cost": 55, "xP": 4.5, "web_name": "FWD5"},
        {"team": 6, "cost": 45, "xP": 3.5, "web_name": "FWD6"},
    ]
    for fw in fwds:
        players.append({"id": pid, "element_type": 4, **fw})
        pid += 1

    return players


def test_solver_standard_budget(mock_player_dataset):
    optimizer = FPLOptimizer()
    result = optimizer.solve(mock_player_dataset, budget=100.0)

    assert isinstance(result, OptimizationResult)
    assert result.status == "Optimal"
    assert len(result.selected_player_ids) == 15
    assert result.total_cost <= 100.0
    assert result.total_projected_points > 0.0

    # Positional constraints check
    assert result.position_counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    # Max 3 per team check
    team_counts: Dict[str, int] = {}
    for p in result.selected_elements:
        t = str(p.get("team"))
        team_counts[t] = team_counts.get(t, 0) + 1
        assert team_counts[t] <= 3, f"Team {t} exceeded max 3 players constraint: {team_counts[t]}"


def test_solver_tight_budget(mock_player_dataset):
    optimizer = FPLOptimizer()
    # Min budget needed for cheapest squad:
    # 2 GKs (4.0+4.5=8.5) + 5 DEFs (4.0*2+4.5*3=21.5) + 5 MIDs (4.5+5.0+5.5+6.0+6.5=27.5) + 3 FWDs (4.5+5.5+6.0=16.0) = 73.5m
    result = optimizer.solve(mock_player_dataset, budget=80.0)

    assert result.status == "Optimal"
    assert len(result.selected_player_ids) == 15
    assert result.total_cost <= 80.0
    assert result.position_counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_solver_infeasible_budget(mock_player_dataset):
    optimizer = FPLOptimizer()
    # Unreasonably low budget (e.g. 30.0m)
    result = optimizer.solve(mock_player_dataset, budget=30.0)

    assert result.status != "Optimal"
    assert len(result.selected_player_ids) == 0


def test_solver_with_player_dataclasses(mock_player_dataset):
    optimizer = FPLOptimizer()
    dataclass_players = []
    for p in mock_player_dataset:
        dataclass_players.append(
            Player(
                id=p["id"],
                web_name=p["web_name"],
                first_name="Test",
                second_name="User",
                team=p["team"],
                element_type=p["element_type"],
                now_cost=p["cost"],
                selected_by_percent="10.0",
                form="5.0",
                points_per_game=str(p["xP"]),
                total_points=int(p["xP"] * 10),
                raw_data=p,
            )
        )

    result = optimizer.solve(dataclass_players, budget=100.0)
    assert result.status == "Optimal"
    assert len(result.selected_player_ids) == 15
    assert result.position_counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
