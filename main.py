#!/usr/bin/env python3
"""
Master Orchestration Script for FPL Manager.
Integrates data ingestion, intelligence, optimization, and team execution.
"""

from typing import Any, Dict, List, Optional
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from src.data_ingestion.fpl_api import FPLDataFetcher
from src.intelligence.perplexity_client import PerplexityNewsClient
from src.intelligence.gemini_fusion import ExpectedPointsGenerator
from src.optimization.ilp_solver import FPLOptimizer
from src.execution.team_manager import FPLExecutor

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure console logging level and format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FPL Manager Automated Pipeline Orchestrator"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run optimization and print proposed squad without making live API mutations (default: True)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Execute live team changes on FPL API",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Total squad budget constraint in millions (default: 100.0)",
    )
    parser.add_argument(
        "--manager-id",
        type=int,
        default=None,
        help="FPL Manager ID (defaults to FPL_MANAGER_ID environment variable)",
    )
    return parser.parse_args(args)


def select_starting_xi(
    squad_players: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    """
    Select valid starting XI (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD) and bench from 15-player squad.
    Appoints captain and vice-captain based on projected points (xP).
    """
    # Group by position
    gks = [p for p in squad_players if p.get("position") == "GK"]
    defs = [p for p in squad_players if p.get("position") == "DEF"]
    mids = [p for p in squad_players if p.get("position") == "MID"]
    fwds = [p for p in squad_players if p.get("position") == "FWD"]

    # Sort each group by xP descending
    for group in (gks, defs, mids, fwds):
        group.sort(key=lambda p: float(p.get("xP", 0.0)), reverse=True)

    # Required base formation: 1 GK, 3 DEF, 2 MID, 1 FWD (7 players)
    starting_xi = [gks[0]] + defs[:3] + mids[:2] + [fwds[0]]

    # Remaining outfield candidates (2 DEFs, 3 MIDs, 2 FWDs)
    remaining_outfield = defs[3:] + mids[2:] + fwds[1:]
    remaining_outfield.sort(key=lambda p: float(p.get("xP", 0.0)), reverse=True)

    # Pick top 4 outfielders to complete XI of 11
    starting_xi.extend(remaining_outfield[:4])

    starting_ids = set(p["id"] for p in starting_xi)
    bench = [p for p in squad_players if p["id"] not in starting_ids]

    # Order bench: substitute GK first, then outfielders by xP descending
    bench_gk = [p for p in bench if p.get("position") == "GK"]
    bench_outfield = [p for p in bench if p.get("position") != "GK"]
    bench_outfield.sort(key=lambda p: float(p.get("xP", 0.0)), reverse=True)
    bench = bench_gk + bench_outfield

    # Sort starting XI by position order (GK, DEF, MID, FWD)
    pos_order = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    starting_xi.sort(key=lambda p: (pos_order.get(p.get("position"), 5), -float(p.get("xP", 0.0))))

    # Captain and Vice-Captain from starting XI
    sorted_xi_by_xp = sorted(starting_xi, key=lambda p: float(p.get("xP", 0.0)), reverse=True)
    captain_id = sorted_xi_by_xp[0]["id"]
    vice_captain_id = sorted_xi_by_xp[1]["id"] if len(sorted_xi_by_xp) > 1 else captain_id

    return starting_xi, bench, captain_id, vice_captain_id


def main(args_list: Optional[List[str]] = None) -> int:
    """Main orchestration workflow."""
    setup_logging()
    load_dotenv()

    args = parse_args(args_list)
    dry_run = not args.live if args.live else args.dry_run
    manager_id = args.manager_id or os.getenv("FPL_MANAGER_ID")

    logger.info(f"Initializing FPL Manager Pipeline (dry_run={dry_run}, budget=£{args.budget}m)...")

    # Step 1: Data Ingestion
    logger.info("Step 1/5: Ingesting live static data from FPL API...")
    fetcher = FPLDataFetcher()
    bootstrap = fetcher.get_bootstrap_data()
    elements = bootstrap.raw_data.get("elements", [])
    logger.info(f"Retrieved {len(elements)} players from FPL bootstrap-static.")

    # Step 2: Perplexity News Search
    logger.info("Step 2/5: Searching qualitative injury & rotation news via Perplexity...")
    top_player_names = [p.get("web_name", "") for p in elements[:15] if p.get("web_name")]
    news_client = PerplexityNewsClient()
    news_context = news_client.fetch_injury_and_rotation_news(top_player_names)
    logger.info(f"News Search complete ({len(news_context.get('summary_text', ''))} chars context).")

    # Step 3: Gemini Fusion Projections
    logger.info("Step 3/5: Generating expected points (xP) forecasts via Gemini Fusion...")
    xp_generator = ExpectedPointsGenerator()
    xp_map = xp_generator.generate_projections(
        bootstrap_data=elements,
        news_context=news_context.get("summary_text"),
    )
    logger.info(f"Generated xP projections for {len(xp_map)} players.")

    # Inject xP into player element dicts
    for player in elements:
        p_id = player.get("id")
        if p_id in xp_map:
            player["xP"] = xp_map[p_id]

    # Step 4: PuLP Squad Optimization
    logger.info(f"Step 4/5: Solving 0-1 ILP squad optimization (budget=£{args.budget}m)...")
    optimizer = FPLOptimizer()
    opt_result = optimizer.solve(elements, budget=args.budget)

    if opt_result.status != "Optimal":
        logger.error(f"Squad optimization failed with status: {opt_result.status}")
        return 1

    logger.info(f"Optimization successful! Total Projected Points: {opt_result.total_projected_points:.2f} xP.")

    # Step 5: Formation Selection & Execution
    logger.info("Step 5/5: Selecting Starting XI, Captaincy, and formatting output...")
    # Parse selected elements to include position code
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    parsed_squad = []
    for p in opt_result.selected_elements:
        p_copy = dict(p)
        p_copy["position"] = pos_map.get(p.get("element_type"), "UNK")
        parsed_squad.append(p_copy)

    starting_xi, bench, captain_id, vice_captain_id = select_starting_xi(parsed_squad)

    executor = FPLExecutor()
    executor.execute_lineup_and_transfers(
        manager_id=manager_id,
        starting_xi=starting_xi,
        bench=bench,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        total_points=opt_result.total_projected_points,
        total_cost=opt_result.total_cost,
        dry_run=dry_run,
    )

    logger.info("FPL Manager Pipeline execution complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
