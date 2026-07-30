"""
FPL Squad Optimization Module using Integer Linear Programming (PuLP).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging
import pulp

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Dataclass holding the optimization result."""
    status: str
    selected_player_ids: List[int]
    selected_elements: List[Dict[str, Any]]
    total_projected_points: float
    total_cost: float
    position_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "selected_player_ids": self.selected_player_ids,
            "selected_elements": self.selected_elements,
            "total_projected_points": round(self.total_projected_points, 2),
            "total_cost": round(self.total_cost, 2),
            "position_counts": self.position_counts,
        }


class FPLOptimizer:
    """
    Integer Linear Programming (ILP) solver for Fantasy Premier League (FPL) squad selection.
    Uses PuLP to optimize squad selection maximizing projected points under FPL rules.
    """

    POSITION_MAP = {
        1: "GK",
        2: "DEF",
        3: "MID",
        4: "FWD",
        "1": "GK",
        "2": "DEF",
        "3": "MID",
        "4": "FWD",
        "GK": "GK",
        "GKP": "GK",
        "DEF": "DEF",
        "MID": "MID",
        "FWD": "FWD",
    }

    REQUIRED_POSITIONS = {
        "GK": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }

    def __init__(self, solver: Optional[pulp.LpSolver] = None):
        self.solver = solver or pulp.PULP_CBC_CMD(msg=False)

    @staticmethod
    def _parse_player(player: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Normalize player dictionary or dataclass into standardized fields."""
        if hasattr(player, "__dict__"):
            data = getattr(player, "raw_data", None) or player.__dict__
        elif isinstance(player, dict):
            data = player
        else:
            raise ValueError(f"Unsupported player object type: {type(player)}")

        # Extract ID
        p_id = data.get("id")
        if p_id is None:
            raise ValueError("Player object missing 'id' attribute")

        # Extract position
        raw_pos = data.get("element_type") or data.get("position")
        pos_str = FPLOptimizer.POSITION_MAP.get(raw_pos)
        if not pos_str:
            raise ValueError(f"Player {p_id} has unknown position/element_type: {raw_pos}")

        # Extract team
        team = data.get("team") or data.get("team_code") or data.get("team_id") or "UNKNOWN"

        # Extract cost (handle now_cost in tenths vs millions)
        raw_cost = data.get("now_cost", data.get("cost", 0.0))
        try:
            cost_val = float(raw_cost)
            cost_millions = cost_val / 10.0 if cost_val > 20.0 else cost_val
        except (ValueError, TypeError):
            cost_millions = 0.0

        # Extract projected points (xP / ep_this / points_per_game / total_points)
        raw_xp = (
            data.get("xP")
            if data.get("xP") is not None
            else data.get("ep_this")
            if data.get("ep_this") is not None
            else data.get("expected_points")
            if data.get("expected_points") is not None
            else data.get("points_per_game")
            if data.get("points_per_game") is not None
            else data.get("total_points", 0.0)
        )
        try:
            xp = float(raw_xp)
        except (ValueError, TypeError):
            xp = 0.0

        return {
            "id": int(p_id),
            "position": pos_str,
            "team": str(team),
            "cost": cost_millions,
            "xP": xp,
            "raw": data,
        }

    def solve(
        self,
        elements: List[Union[Dict[str, Any], Any]],
        budget: float = 100.0,
        current_squad_ids: Optional[List[int]] = None,
        max_players_per_team: int = 3,
    ) -> OptimizationResult:
        """
        Formulate and solve 0-1 ILP squad selection problem.

        Args:
            elements: List of player dicts or objects.
            budget: Total squad budget constraint (in millions, e.g., 100.0).
            current_squad_ids: Optional list of current player IDs (for future transfer extensions).
            max_players_per_team: Max allowed players per team (default 3).

        Returns:
            OptimizationResult containing optimal selection, total points, total cost, and status.
        """
        parsed_players = [self._parse_player(p) for p in elements]
        if not parsed_players:
            return OptimizationResult(
                status="Infeasible",
                selected_player_ids=[],
                selected_elements=[],
                total_projected_points=0.0,
                total_cost=0.0,
                position_counts={},
            )

        # Create PuLP Problem
        prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)

        # Decision Variables: x_i in {0, 1}
        x_vars = {
            p["id"]: pulp.LpVariable(f"x_{p['id']}", cat=pulp.LpBinary)
            for p in parsed_players
        }

        # Objective Function: Maximize sum(xP_i * x_i)
        prob += (
            pulp.lpSum([p["xP"] * x_vars[p["id"]] for p in parsed_players]),
            "Total_Projected_Points",
        )

        # Constraint 1: Total squad size = 15
        prob += (
            pulp.lpSum([x_vars[p["id"]] for p in parsed_players]) == 15,
            "Squad_Size_15",
        )

        # Constraint 2: Positional constraints (2 GK, 5 DEF, 5 MID, 3 FWD)
        for pos_code, count in self.REQUIRED_POSITIONS.items():
            pos_players = [p for p in parsed_players if p["position"] == pos_code]
            prob += (
                pulp.lpSum([x_vars[p["id"]] for p in pos_players]) == count,
                f"Pos_Count_{pos_code}",
            )

        # Constraint 3: Max N players per team
        teams = set(p["team"] for p in parsed_players)
        for team in teams:
            team_players = [p for p in parsed_players if p["team"] == team]
            prob += (
                pulp.lpSum([x_vars[p["id"]] for p in team_players]) <= max_players_per_team,
                f"Max_Team_{team}",
            )

        # Constraint 4: Budget constraint
        prob += (
            pulp.lpSum([p["cost"] * x_vars[p["id"]] for p in parsed_players]) <= budget,
            "Budget_Constraint",
        )

        # Solve model
        prob.solve(self.solver)
        status_str = pulp.LpStatus[prob.status]

        if status_str != "Optimal":
            logger.warning(f"Optimization finished with non-optimal status: {status_str}")
            return OptimizationResult(
                status=status_str,
                selected_player_ids=[],
                selected_elements=[],
                total_projected_points=0.0,
                total_cost=0.0,
                position_counts={},
            )

        # Collect selected elements
        selected_players = [
            p for p in parsed_players if pulp.value(x_vars[p["id"]]) == 1
        ]
        selected_ids = [p["id"] for p in selected_players]
        selected_raws = [p["raw"] for p in selected_players]
        total_xp = sum(p["xP"] for p in selected_players)
        total_cost = sum(p["cost"] for p in selected_players)

        pos_counts: Dict[str, int] = {}
        for p in selected_players:
            pos_counts[p["position"]] = pos_counts.get(p["position"], 0) + 1

        return OptimizationResult(
            status=status_str,
            selected_player_ids=selected_ids,
            selected_elements=selected_raws,
            total_projected_points=total_xp,
            total_cost=total_cost,
            position_counts=pos_counts,
        )
