"""
Perplexity API Integration Module for FPL Injury, Press Conference, and Rotation News.
"""

from typing import Any, Dict, List, Optional
import logging
import os
import requests

logger = logging.getLogger(__name__)


class PerplexityNewsClient:
    """
    Client for fetching real-time news, press conference quotes, injury status,
    and rotation risks using Perplexity Sonar search models.
    """

    API_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.timeout = timeout

    def _fallback_response(self, player_names: List[str]) -> Dict[str, Any]:
        """Construct default baseline response when API key is missing or request fails."""
        results: Dict[str, Any] = {}
        for player in player_names:
            results[player] = {
                "status": "Available",
                "news": "No critical injury or rotation flags reported.",
                "risk_multiplier": 1.0,
            }
        results["summary_text"] = "No external news flags detected (fallback mode active)."
        return results

    def fetch_injury_and_rotation_news(
        self,
        player_names: List[str],
        model: str = "sonar-pro",
    ) -> Dict[str, Any]:
        """
        Fetch latest press conference quotes, injury news, and rotation risks for specified players.

        Args:
            player_names: List of player web names or full names to query.
            model: Perplexity model identifier (default: "sonar-pro").

        Returns:
            Dictionary containing qualitative news breakdown per player and summary text.
        """
        if not player_names:
            return self._fallback_response([])

        if not self.api_key:
            logger.info("PERPLEXITY_API_KEY not configured. Returning fallback news context.")
            return self._fallback_response(player_names)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        players_formatted = ", ".join(player_names)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Fantasy Premier League (FPL) news aggregator. "
                    "Provide concise, factual updates regarding recent press conference quotes, "
                    "injuries, fitness concerns, and rotation risks for the requested players."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Check latest team news and injury/rotation risk for: {players_formatted}. "
                    "For each player, briefly state fitness status and potential risk."
                ),
            },
        ]

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Build result dictionary
            results: Dict[str, Any] = {"summary_text": content}
            for player in player_names:
                # Default entry unless player mentioned in content with specific flag
                results[player] = {
                    "status": "Check News",
                    "news": content,
                    "risk_multiplier": 1.0,
                }

            return results

        except requests.exceptions.RequestException as exc:
            logger.warning(
                f"Perplexity API request failed ({exc}). Using fallback news context."
            )
            return self._fallback_response(player_names)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            logger.warning(
                f"Failed to parse Perplexity API response ({exc}). Using fallback news context."
            )
            return self._fallback_response(player_names)
