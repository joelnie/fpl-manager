"""
FPL API Client Module for fetching data from the Fantasy Premier League API.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

logger = logging.getLogger(__name__)


class FPLAPIError(Exception):
    """Custom exception raised when FPL API requests or response parsing fail."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


@dataclass
class Player:
    """Dataclass representing an FPL player (element)."""
    id: int
    web_name: str
    first_name: str
    second_name: str
    team: int
    element_type: int
    now_cost: int
    selected_by_percent: str
    form: str
    points_per_game: str
    total_points: int
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Player":
        return cls(
            id=data.get("id", 0),
            web_name=data.get("web_name", ""),
            first_name=data.get("first_name", ""),
            second_name=data.get("second_name", ""),
            team=data.get("team", 0),
            element_type=data.get("element_type", 0),
            now_cost=data.get("now_cost", 0),
            selected_by_percent=data.get("selected_by_percent", "0.0"),
            form=data.get("form", "0.0"),
            points_per_game=data.get("points_per_game", "0.0"),
            total_points=data.get("total_points", 0),
            raw_data=data,
        )


@dataclass
class Team:
    """Dataclass representing an FPL team."""
    id: int
    name: str
    short_name: str
    code: int
    strength: int
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Team":
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            short_name=data.get("short_name", ""),
            code=data.get("code", 0),
            strength=data.get("strength", 0),
            raw_data=data,
        )


@dataclass
class Gameweek:
    """Dataclass representing an FPL gameweek (event)."""
    id: int
    name: str
    deadline_time: str
    is_previous: bool
    is_current: bool
    is_next: bool
    finished: bool
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Gameweek":
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            deadline_time=data.get("deadline_time", ""),
            is_previous=data.get("is_previous", False),
            is_current=data.get("is_current", False),
            is_next=data.get("is_next", False),
            finished=data.get("finished", False),
            raw_data=data,
        )


@dataclass
class BootstrapData:
    """Dataclass representing static bootstrap data from FPL API."""
    elements: List[Player]
    teams: List[Team]
    events: List[Gameweek]
    element_types: List[Dict[str, Any]]
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BootstrapData":
        elements = [Player.from_dict(el) for el in data.get("elements", [])]
        teams = [Team.from_dict(tm) for tm in data.get("teams", [])]
        events = [Gameweek.from_dict(ev) for ev in data.get("events", [])]
        element_types = data.get("element_types", [])
        return cls(
            elements=elements,
            teams=teams,
            events=events,
            element_types=element_types,
            raw_data=data,
        )

    def to_player_df(self) -> pd.DataFrame:
        if self.elements:
            return pd.DataFrame([e.raw_data for e in self.elements])
        return pd.DataFrame()

    def to_team_df(self) -> pd.DataFrame:
        if self.teams:
            return pd.DataFrame([t.raw_data for t in self.teams])
        return pd.DataFrame()

    def to_gameweek_df(self) -> pd.DataFrame:
        if self.events:
            return pd.DataFrame([g.raw_data for g in self.events])
        return pd.DataFrame()


@dataclass
class Fixture:
    """Dataclass representing an FPL match fixture."""
    id: int
    code: int
    team_h: int
    team_a: int
    team_h_score: Optional[int]
    team_a_score: Optional[int]
    event: Optional[int]
    finished: bool
    kickoff_time: Optional[str]
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fixture":
        return cls(
            id=data.get("id", 0),
            code=data.get("code", 0),
            team_h=data.get("team_h", 0),
            team_a=data.get("team_a", 0),
            team_h_score=data.get("team_h_score"),
            team_a_score=data.get("team_a_score"),
            event=data.get("event"),
            finished=data.get("finished", False),
            kickoff_time=data.get("kickoff_time"),
            raw_data=data,
        )


@dataclass
class PlayerSummary:
    """Dataclass representing detailed stats and fixtures for a specific player."""
    player_id: int
    fixtures: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    history_past: List[Dict[str, Any]]
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, player_id: int, data: Dict[str, Any]) -> "PlayerSummary":
        return cls(
            player_id=player_id,
            fixtures=data.get("fixtures", []),
            history=data.get("history", []),
            history_past=data.get("history_past", []),
            raw_data=data,
        )

    def to_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def to_fixtures_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.fixtures)


@dataclass
class MyTeamPick:
    """Dataclass representing a player pick in a manager's team."""
    element: int
    position: int
    selling_price: int
    purchase_price: int
    multiplier: int
    is_captain: bool
    is_vice_captain: bool
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MyTeamPick":
        return cls(
            element=data.get("element", 0),
            position=data.get("position", 0),
            selling_price=data.get("selling_price", 0),
            purchase_price=data.get("purchase_price", 0),
            multiplier=data.get("multiplier", 1),
            is_captain=data.get("is_captain", False),
            is_vice_captain=data.get("is_vice_captain", False),
            raw_data=data,
        )


@dataclass
class MyTeam:
    """Dataclass representing manager team information."""
    user_id: int
    picks: List[MyTeamPick]
    chips: List[Dict[str, Any]]
    transfers: Dict[str, Any]
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, user_id: int, data: Dict[str, Any]) -> "MyTeam":
        picks = [MyTeamPick.from_dict(p) for p in data.get("picks", [])]
        chips = data.get("chips", [])
        transfers = data.get("transfers", {})
        return cls(
            user_id=user_id,
            picks=picks,
            chips=chips,
            transfers=transfers,
            raw_data=data,
        )

    def to_picks_df(self) -> pd.DataFrame:
        if self.picks:
            return pd.DataFrame([p.raw_data for p in self.picks])
        return pd.DataFrame()


class FPLDataFetcher:
    """Client for interacting with Fantasy Premier League (FPL) endpoints."""

    BASE_URL = "https://fantasy.premierleague.com/api"
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        session_cookies: Optional[Dict[str, str]] = None,
    ):
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)

        if session_cookies:
            self.session.cookies.update(session_cookies)

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request(
        self,
        endpoint: str,
        cookies: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req_headers = dict(headers) if headers else None

        try:
            response = self.session.get(
                url,
                timeout=self.timeout,
                cookies=cookies,
                headers=req_headers,
            )
        except requests.exceptions.Timeout as exc:
            logger.error(f"Timeout occurred while requesting {url}: {exc}")
            raise FPLAPIError(f"Request timeout after {self.timeout}s: {url}") from exc
        except requests.exceptions.RequestException as exc:
            logger.error(f"Network error while requesting {url}: {exc}")
            raise FPLAPIError(f"Network error: {url}") from exc

        if not response.ok:
            logger.error(f"HTTP error {response.status_code} requesting {url}")
            raise FPLAPIError(
                f"HTTP {response.status_code} Error: {url}",
                status_code=response.status_code,
                response=response,
            )

        try:
            return response.json()
        except ValueError as exc:
            logger.error(f"Invalid JSON response from {url}: {exc}")
            raise FPLAPIError(f"Failed to parse JSON response from {url}") from exc

    def get_bootstrap_data(self) -> BootstrapData:
        """Fetch general FPL data including players, teams, gameweeks, and element types."""
        raw_data = self._request("bootstrap-static/")
        return BootstrapData.from_dict(raw_data)

    def get_fixtures(self, event: Optional[int] = None) -> List[Fixture]:
        """
        Fetch match fixtures.
        
        Args:
            event: Optional gameweek number. If provided, fetches fixtures for that event (e.g. fixtures/?event=1).
        """
        endpoint = f"fixtures/?event={event}" if event is not None else "fixtures/"
        raw_data = self._request(endpoint)
        if not isinstance(raw_data, list):
            raise FPLAPIError("Expected list response for fixtures endpoint")
        return [Fixture.from_dict(f) for f in raw_data]

    def get_player_summary(self, player_id: int) -> PlayerSummary:
        """Fetch detailed stats, history, and upcoming fixtures for a given player ID."""
        endpoint = f"element-summary/{player_id}/"
        raw_data = self._request(endpoint)
        return PlayerSummary.from_dict(player_id, raw_data)

    def get_my_team(
        self,
        manager_id: Optional[int] = None,
        user_id: Optional[int] = None,
        session_cookie: Optional[Dict[str, str]] = None,
        session_cookies: Optional[Dict[str, str]] = None,
    ) -> MyTeam:
        """
        Fetch team picks and manager data for authenticated user/manager.
        Supports both manager_id and user_id parameter names, as well as session_cookie / session_cookies.
        Gracefully handles session headers and credentials.
        """
        target_id = manager_id if manager_id is not None else user_id
        if target_id is None:
            raise FPLAPIError("Must provide either manager_id or user_id to get_my_team()")

        cookies = session_cookie or session_cookies
        endpoint = f"my-team/{target_id}/"
        active_cookies = cookies or dict(self.session.cookies)

        if not active_cookies:
            logger.warning(
                f"Attempting unauthenticated access to my-team endpoint for manager {target_id}. "
                "FPL API requires session cookies (pl_profile)."
            )

        try:
            raw_data = self._request(endpoint, cookies=cookies)
        except FPLAPIError as err:
            if err.status_code in (401, 403):
                raise FPLAPIError(
                    f"Authentication required for my-team/{target_id}/. "
                    "Provide valid session_cookies (e.g., pl_profile).",
                    status_code=err.status_code,
                    response=err.response,
                ) from err
            raise

        return MyTeam.from_dict(target_id, raw_data)
