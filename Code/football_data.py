"""Client for retrieving football matches from footballdata.io."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

import dotenv
import requests

from storage import FixtureStorage


class FootballDataClient:
    """Fetch match data from the footballdata.io API."""

    BASE_URL = "https://footballdata.io/api/v1"
    TARGET_LEAGUE = "England Premier League"
    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: requests.Session | None = None,
        base_url: str = BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        dotenv.load_dotenv()
        self.api_key = api_key or os.getenv("FOOTBALLDATAIO_KEY")
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_matches(self, match_date: dt.date | None = None) -> list[dict[str, Any]]:
        """Return England Premier League matches for a date (yesterday by default)."""
        if not self.api_key:
            raise ValueError("FOOTBALLDATAIO_KEY is not configured")

        requested_date = match_date or (dt.date.today() - dt.timedelta(days=1))
        url = f"{self.base_url}/matches/date/{requested_date.isoformat()}"
        response = self.session.get(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        matches = data.get("matches", []) if isinstance(data, dict) else []
        if not isinstance(matches, list):
            return []

        return [
            match
            for match in matches
            if isinstance(match, dict) and self._is_target_league(match)
        ]

    @classmethod
    def _is_target_league(cls, match: dict[str, Any]) -> bool:
        league = match.get("league", "")
        league_name = league.get("name", "") if isinstance(league, dict) else league
        return league_name == cls.TARGET_LEAGUE


def _value(match: dict[str, Any], *keys: str) -> Any:
    """Return the first present value, including values nested in common fields."""
    for key in keys:
        if key in match and match[key] not in (None, ""):
            return match[key]
    return ""


def format_match(match: dict[str, Any]) -> str:
    """Format one API match as a compact, readable line."""
    league = match.get("league", "Unknown league")
    league = league.get("name", "Unknown league") if isinstance(league, dict) else league
    teams = match.get("teams", {})
    teams = teams if isinstance(teams, dict) else {}
    home = match.get(
        "home_team",
        match.get("homeTeam", match.get("home", teams.get("home", "?"))),
    )
    away = match.get(
        "away_team",
        match.get("awayTeam", match.get("away", teams.get("away", "?"))),
    )
    home = home.get("team_name", home.get("name", "?")) if isinstance(home, dict) else home
    away = away.get("team_name", away.get("name", "?")) if isinstance(away, dict) else away
    score = match.get("score", {})
    if isinstance(score, dict):
        score = f"{_value(score, 'home', 'home_score')}–{_value(score, 'away', 'away_score')}"
    score = score or f"{_value(match, 'home_score', 'homeScore')}–{_value(match, 'away_score', 'awayScore')}"
    kickoff = _value(match, "date", "match_date", "kickoff", "start_time")
    return f"{league} | {kickoff or 'time unknown'} | {home} {score} {away}"


if __name__ == "__main__":
    client = FootballDataClient()
    storage = FixtureStorage(Path(__file__).with_name("data.csv"))
    yesterday = dt.date.today() - dt.timedelta(days=1)
    try:
        matches = client.get_matches(yesterday)
    except requests.exceptions.RequestException as error:
        print(f"Could not retrieve match data: {error}")
    else:
        storage.save_matchday(matches)
        stored_matches = storage.load_matchday()
        print(f"{client.TARGET_LEAGUE} matches for {yesterday.isoformat()}: {len(stored_matches)}")
        for match in stored_matches.to_dict("records"):
            print(
                f"{match['league']} | {match['match_date']} | "
                f"{match['home_team']} {match['home_score']}–{match['away_score']} {match['away_team']}"
            )
