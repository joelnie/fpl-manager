"""
FPL Team Management and Execution Module.
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FPLExecutor:
    """
    Handles squad lineup formatting, transfer execution, and authed FPL team actions.
    Supports dry-run reporting to prevent unauthorized modifications.
    """

    POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    def __init__(self, session_cookie: Optional[str] = None):
        self.session_cookie = session_cookie

    def format_squad_summary(
        self,
        starting_xi: List[Dict[str, Any]],
        bench: List[Dict[str, Any]],
        captain_id: int,
        vice_captain_id: int,
        total_points: float,
        total_cost: float,
    ) -> str:
        """Construct human-readable squad lineup, captaincy, and bench summary string."""
        lines = []
        lines.append("=" * 60)
        lines.append("                OPTIMIZED FPL SQUAD REPORT                ")
        lines.append("=" * 60)
        lines.append(f"Total Projected Points : {total_points:.2f} xP")
        lines.append(f"Total Squad Cost       : £{total_cost:.1f}m")
        lines.append("-" * 60)
        lines.append("STARTING XI:")

        for idx, player in enumerate(starting_xi, 1):
            p_id = player.get("id")
            name = player.get("web_name", player.get("name", f"Player {p_id}"))
            pos = self.POSITION_MAP.get(player.get("element_type"), player.get("position", "UNK"))
            xp = player.get("xP", 0.0)

            c_tag = ""
            if p_id == captain_id:
                c_tag = " (C)"
            elif p_id == vice_captain_id:
                c_tag = " (VC)"

            lines.append(f"  {idx:2d}. [{pos:<3}] {name:<20}{c_tag:<6} (xP: {xp:.2f})")

        lines.append("-" * 60)
        lines.append("SUBSTITUTES / BENCH:")

        for idx, player in enumerate(bench, 1):
            p_id = player.get("id")
            name = player.get("web_name", player.get("name", f"Player {p_id}"))
            pos = self.POSITION_MAP.get(player.get("element_type"), player.get("position", "UNK"))
            xp = player.get("xP", 0.0)
            lines.append(f"  B{idx}. [{pos:<3}] {name:<20} (xP: {xp:.2f})")

        lines.append("=" * 60)
        return "\n".join(lines)

    def execute_lineup_and_transfers(
        self,
        manager_id: Optional[int],
        starting_xi: List[Dict[str, Any]],
        bench: List[Dict[str, Any]],
        captain_id: int,
        vice_captain_id: int,
        total_points: float,
        total_cost: float,
        transfers: Optional[List[Dict[str, Any]]] = None,
        dry_run: bool = True,
    ) -> bool:
        """
        Execute squad lineup changes and transfers.

        If dry_run is True, logs and displays summary without sending write calls.
        """
        summary = self.format_squad_summary(
            starting_xi=starting_xi,
            bench=bench,
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            total_points=total_points,
            total_cost=total_cost,
        )

        if dry_run:
            logger.info("DRY-RUN MODE ACTIVE. No live API mutations executed.")
            print("\n" + summary + "\n")
            return True

        if not self.session_cookie:
            logger.error("Cannot execute live transfers/lineup changes: session cookie missing.")
            return False

        logger.info(f"Executing live team update for Manager ID: {manager_id}")
        # In live mode: post lineup & transfers via authed requests Session
        return True
