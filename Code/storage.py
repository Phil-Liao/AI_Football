"""Thread/process-safe CSV storage for matchday fixtures."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import pandas as pd


class FixtureStorage:
    """Store and retrieve fixtures in a CSV file using Pandas.

    A process-level lock protects read/modify/write operations. Writes go to a
    temporary file and are committed with ``os.replace`` so readers never see
    a partially-written CSV.
    """

    COLUMNS = [
        "match_id", "match_date", "league", "home_team", "away_team",
        "home_score", "away_score", "status", "raw_match",
    ]

    def __init__(self, path: str | Path = "data.csv") -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    @contextmanager
    def _lock(self) -> Generator[None, None, None]:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def save_matchday(self, matches: list[dict[str, Any]]) -> pd.DataFrame:
        """Replace the stored matchday with ``matches`` and return the frame."""
        frame = pd.DataFrame([self._normalize(match) for match in matches], columns=self.COLUMNS)
        with self._lock():
            self._atomic_write(frame)
        return frame

    def load_matchday(self) -> pd.DataFrame:
        """Load the stored matchday, returning an empty typed frame if absent."""
        with self._lock():
            if not self.path.exists():
                return pd.DataFrame(columns=self.COLUMNS)
            frame = pd.read_csv(self.path, dtype={"match_id": "string"})
        return frame.reindex(columns=self.COLUMNS)

    def _atomic_write(self, frame: pd.DataFrame) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            frame.to_csv(temporary_path, index=False)
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _normalize(match: dict[str, Any]) -> dict[str, Any]:
        league = match.get("league", {})
        home = match.get("home_team", {})
        away = match.get("away_team", {})
        score = match.get("score", {})
        return {
            "match_id": match.get("match_id", match.get("id", "")),
            "match_date": match.get("match_date", match.get("date", "")),
            "league": league.get("name", "") if isinstance(league, dict) else league,
            "home_team": home.get("team_name", home.get("name", "")) if isinstance(home, dict) else home,
            "away_team": away.get("team_name", away.get("name", "")) if isinstance(away, dict) else away,
            "home_score": score.get("home", "") if isinstance(score, dict) else "",
            "away_score": score.get("away", "") if isinstance(score, dict) else "",
            "status": match.get("status", ""),
            "raw_match": json.dumps(match, separators=(",", ":"), default=str),
        }
