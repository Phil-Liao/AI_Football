"""Thread/process-safe CSV storage for fixtures and predictions."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


class CsvStorage:
    """Base class for race-safe CSV read and atomic read-modify-write operations."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._thread_lock = _thread_lock(self.path)

    @contextmanager
    def _lock(self) -> Generator[None, None, None]:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            with self.lock_path.open("a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

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

    def _read_csv(self, columns: list[str]) -> pd.DataFrame:
        try:
            if not self.path.exists() or self.path.stat().st_size == 0:
                return pd.DataFrame(columns=columns)

            frame = pd.read_csv(self.path, dtype={"match_id": "string"})
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            OSError,
        ):
            return pd.DataFrame(columns=columns)
        return frame.reindex(columns=columns)


class FixtureStorage(CsvStorage):
    """Store and retrieve Premier League fixtures."""

    COLUMNS = [
        "match_id", "match_date", "league", "home_team", "away_team",
        "home_score", "away_score", "status", "raw_match",
    ]

    def __init__(self, path: str | Path | None = None) -> None:
        fixture_path = Path(path) if path is not None else Path(__file__).with_name("data.csv")
        super().__init__(fixture_path)

    def save_matchday(self, matches: list[dict[str, Any]]) -> pd.DataFrame:
        """Replace the stored fixture list with ``matches``."""
        frame = pd.DataFrame([self._normalize(match) for match in matches], columns=self.COLUMNS)
        with self._lock():
            self._atomic_write(frame)
        return frame

    def merge_matchday(self, matches: list[dict[str, Any]]) -> pd.DataFrame:
        """Replace matching fixture IDs and preserve other stored fixtures."""
        incoming = pd.DataFrame([self._normalize(match) for match in matches], columns=self.COLUMNS)
        with self._lock():
            existing = self._load_frame_unlocked()
            if incoming.empty:
                return existing

            incoming_ids = incoming["match_id"].astype("string").fillna("")
            if not existing.empty:
                existing_ids = existing["match_id"].astype("string").fillna("")
                existing = existing.loc[~existing_ids.isin(incoming_ids)].copy()

            combined = pd.concat([existing, incoming], ignore_index=True)
            combined = combined.reindex(columns=self.COLUMNS)
            self._atomic_write(combined)
            return combined

    def load_matchday(self) -> pd.DataFrame:
        """Load fixtures without exposing partial CSV writes."""
        with self._lock():
            return self._load_frame_unlocked()

    def _load_frame_unlocked(self) -> pd.DataFrame:
        return self._read_csv(self.COLUMNS)

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


class PredictionStorage(CsvStorage):
    """Store model predictions with match/model/prompt upsert semantics."""

    COLUMNS = [
        "match_id", "match_date", "home_team", "away_team", "model",
        "prediction", "prompt_version", "status", "created_at",
    ]

    def __init__(self, path: str | Path | None = None) -> None:
        prediction_path = Path(path) if path is not None else Path(__file__).with_name("predictions.csv")
        super().__init__(prediction_path)

    def save_prediction(self, prediction: dict[str, Any]) -> pd.DataFrame:
        return self.save_predictions([prediction])

    def save_predictions(self, predictions: list[dict[str, Any]]) -> pd.DataFrame:
        incoming = pd.DataFrame([self._normalize(row) for row in predictions], columns=self.COLUMNS)
        if incoming.empty:
            return self.load_predictions()

        keys = ["match_id", "model", "prompt_version"]
        with self._lock():
            existing = self._load_frame_unlocked()
            if not existing.empty:
                existing_keys = existing[keys].astype("string").fillna("").agg("\x1f".join, axis=1)
                incoming_keys = incoming[keys].astype("string").fillna("").agg("\x1f".join, axis=1)
                existing = existing.loc[~existing_keys.isin(incoming_keys)].copy()

            combined = pd.concat([existing, incoming], ignore_index=True).reindex(columns=self.COLUMNS)
            self._atomic_write(combined)
            return combined

    def load_predictions(self) -> pd.DataFrame:
        with self._lock():
            return self._load_frame_unlocked()

    def _load_frame_unlocked(self) -> pd.DataFrame:
        return self._read_csv(self.COLUMNS)

    @staticmethod
    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "match_id": str(row.get("match_id", "")),
            "match_date": row.get("match_date", ""),
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "model": row.get("model", ""),
            "prediction": row.get("prediction", ""),
            "prompt_version": str(row.get("prompt_version", "")),
            "status": row.get("status", "stored"),
            "created_at": row.get("created_at") or datetime.now(timezone.utc).isoformat(),
        }


class ScoreStorage(CsvStorage):
    """Store one immutable score row per match/model/prompt combination."""

    COLUMNS = [
        "match_id", "match_date", "home_team", "away_team", "model",
        "prompt_version", "prediction", "actual_outcome", "points",
        "status", "scored_at",
    ]

    def __init__(self, path: str | Path | None = None) -> None:
        score_path = Path(path) if path is not None else Path(__file__).with_name("scores.csv")
        super().__init__(score_path)

    def save_scores(self, scores: list[dict[str, Any]]) -> pd.DataFrame:
        incoming = pd.DataFrame([self._normalize(row) for row in scores], columns=self.COLUMNS)
        if incoming.empty:
            return self.load_scores()

        keys = ["match_id", "model", "prompt_version"]
        with self._lock():
            existing = self._load_frame_unlocked()
            if not existing.empty:
                existing_keys = existing[keys].astype("string").fillna("").agg("\x1f".join, axis=1)
                incoming_keys = incoming[keys].astype("string").fillna("").agg("\x1f".join, axis=1)
                existing = existing.loc[~existing_keys.isin(incoming_keys)].copy()

            combined = pd.concat([existing, incoming], ignore_index=True).reindex(columns=self.COLUMNS)
            self._atomic_write(combined)
            return combined

    def load_scores(self) -> pd.DataFrame:
        with self._lock():
            return self._load_frame_unlocked()

    def _load_frame_unlocked(self) -> pd.DataFrame:
        return self._read_csv(self.COLUMNS)

    @staticmethod
    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "match_id": str(row.get("match_id", "")),
            "match_date": row.get("match_date", ""),
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "model": row.get("model", ""),
            "prompt_version": str(row.get("prompt_version", "")),
            "prediction": row.get("prediction", ""),
            "actual_outcome": row.get("actual_outcome", ""),
            "points": int(row.get("points", 0)),
            "status": row.get("status", "scored"),
            "scored_at": row.get("scored_at") or datetime.now(timezone.utc).isoformat(),
        }
