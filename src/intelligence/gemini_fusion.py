"""
Gemini Intelligence Fusion Module for FPL Expected Points Projections.
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging
import os

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class PlayerProjection(BaseModel):
    """Structured Pydantic schema for individual player point projection."""
    player_id: int = Field(description="FPL player element ID")
    expected_points: float = Field(description="Projected expected points for the next gameweek")
    confidence_score: float = Field(default=0.8, description="Confidence score between 0.0 and 1.0")
    rationale: str = Field(default="", description="Brief explanation for the projected score")


class PlayerProjectionsResponse(BaseModel):
    """Container schema for structured Gemini API response."""
    projections: List[PlayerProjection]


class ExpectedPointsGenerator:
    """
    Projection generator leveraging Google Gemini LLM models for multi-factor FPL point forecasts.
    Combines statistical form, ICT index, fixture difficulty, and news context into structured projections.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client: Optional[genai.Client] = None

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as err:
                logger.warning(f"Failed to initialize Gemini Client with provided API key: {err}")
        else:
            try:
                # Fall back to default environment client initialization if available
                self.client = genai.Client()
            except Exception:
                logger.info("GEMINI_API_KEY not provided. Fallback projection logic will be used.")

    @staticmethod
    def _extract_players(bootstrap_data: Any) -> List[Dict[str, Any]]:
        """Normalize bootstrap_data (BootstrapData, dict, or list of player objects) into player dicts."""
        if hasattr(bootstrap_data, "elements"):
            raw_elements = getattr(bootstrap_data, "elements", [])
        elif isinstance(bootstrap_data, dict):
            raw_elements = bootstrap_data.get("elements", [])
        elif isinstance(bootstrap_data, list):
            raw_elements = bootstrap_data
        else:
            raw_elements = []

        players = []
        for p in raw_elements:
            if hasattr(p, "raw_data") and isinstance(getattr(p, "raw_data"), dict):
                players.append(p.raw_data)
            elif hasattr(p, "__dict__"):
                players.append(p.__dict__)
            elif isinstance(p, dict):
                players.append(p)
        return players

    def _fallback_projections(self, players: List[Dict[str, Any]]) -> Dict[int, float]:
        """
        Baseline fallback calculation when Gemini API key is absent or request fails.
        Calculates xP from player's ep_next, form, or points_per_game.
        """
        projections: Dict[int, float] = {}
        for p in players:
            p_id = p.get("id")
            if p_id is None:
                continue

            raw_xp = (
                p.get("ep_next")
                if p.get("ep_next") is not None
                else p.get("ep_this")
                if p.get("ep_this") is not None
                else p.get("form")
                if p.get("form") is not None
                else p.get("points_per_game", 0.0)
            )

            try:
                xp = float(raw_xp)
            except (ValueError, TypeError):
                xp = 0.0

            projections[int(p_id)] = max(0.0, xp)
        return projections

    def generate_projections(
        self,
        bootstrap_data: Any,
        news_context: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ) -> Dict[int, float]:
        """
        Generate projected expected points (xP) for FPL players.

        Args:
            bootstrap_data: FPL static bootstrap data or player list.
            news_context: Optional additional news/injury context text (e.g. from Perplexity).
            model: Gemini model identifier (default: "gemini-2.5-flash").

        Returns:
            Dictionary mapping player_id -> expected_points float.
        """
        players = self._extract_players(bootstrap_data)
        if not players:
            return {}

        if not self.client:
            logger.info("Using baseline fallback expected points projections (no Gemini Client).")
            return self._fallback_projections(players)

        # Prepare input player summary snippet (top candidate players or subset to avoid token bloat)
        player_summaries = []
        for p in players:
            player_summaries.append(
                {
                    "id": p.get("id"),
                    "name": p.get("web_name", p.get("first_name", "")),
                    "form": p.get("form", "0.0"),
                    "ict_index": p.get("ict_index", "0.0"),
                    "points_per_game": p.get("points_per_game", "0.0"),
                    "ep_next": p.get("ep_next", "0.0"),
                }
            )

        prompt = (
            "You are an expert FPL AI assistant. Analyze the given player performance metrics "
            "and external news context to predict expected points (xP) for the upcoming gameweek.\n\n"
            f"Player Data Summary:\n{json.dumps(player_summaries, indent=2)}\n\n"
        )
        if news_context:
            prompt += f"Recent News & Injury Context:\n{news_context}\n\n"

        prompt += "Return a structured list of expected point predictions for each player ID."

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PlayerProjectionsResponse,
            temperature=0.2,
        )

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            response_text = response.text or ""
            parsed = PlayerProjectionsResponse.model_validate_json(response_text)

            projections = {
                proj.player_id: max(0.0, float(proj.expected_points))
                for proj in parsed.projections
            }

            # Fill in any missing player IDs with fallback logic
            fallback = self._fallback_projections(players)
            for p_id, xp in fallback.items():
                if p_id not in projections:
                    projections[p_id] = xp

            return projections

        except Exception as err:
            logger.warning(
                f"Gemini API generation failed ({err}). Falling back to baseline projections."
            )
            return self._fallback_projections(players)
