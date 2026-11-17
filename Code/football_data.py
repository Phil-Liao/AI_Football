"""Client for retrieving Premier League results from footballdata.io."""

from __future__ import annotations

import datetime as dt
import os
from typing import Any

import dotenv
import requests


class FootballDataClient:
    """Fetch match data from the footballdata.io API."""

    BASE_URL = "https://footballdata.io/api/v1"
    PREMIER_LEAGUE_ID = 15

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
        """Return Premier League matches for a date (yesterday by default)."""
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
            if isinstance(match, dict)
            and self._is_premier_league_match(match)
        ]

    @classmethod
    def _is_premier_league_match(cls, match: dict[str, Any]) -> bool:
        """Check common league ID fields used by football data responses."""
        for key in ("league_id", "leagueId"):
            if match.get(key) == cls.PREMIER_LEAGUE_ID:
                return True

        league = match.get("league")
        return isinstance(league, dict) and league.get("id") == cls.PREMIER_LEAGUE_ID


if __name__ == "__main__":
    client = FootballDataClient()
    yesterday = dt.date.today() - dt.timedelta(days=1)
    try:
        matches = client.get_matches(yesterday)
    except requests.exceptions.RequestException as error:
        print(f"Could not retrieve match data: {error}")
    else:
        print(f"Premier League matches for {yesterday.isoformat()}: {len(matches)}")
        for match in matches:
            print(match)
