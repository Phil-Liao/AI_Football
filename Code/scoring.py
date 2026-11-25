"""Match prediction scoring and leaderboard aggregation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from storage import PredictionStorage, ScoreStorage


COMPLETED_STATUSES = {"complete", "completed", "finished", "ft"}


class PredictionScorer:
    """Score stored predictions against completed fixture outcomes."""

    def __init__(
        self,
        prediction_storage: PredictionStorage | None = None,
        score_storage: ScoreStorage | None = None,
    ) -> None:
        self.prediction_storage = prediction_storage or PredictionStorage()
        self.score_storage = score_storage or ScoreStorage()

    def score_completed_fixtures(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Score all available prediction rows for final completed fixtures."""
        if fixtures.empty:
            return self.score_storage.load_scores()

        predictions = self.prediction_storage.load_predictions()
        if predictions.empty:
            return self.score_storage.load_scores()

        fixture_records = fixtures.to_dict("records")
        prediction_records = predictions.to_dict("records")
        fixtures_by_id = {
            str(row.get("match_id", "")): row
            for row in fixture_records
            if str(row.get("match_id", ""))
        }

        scores: list[dict[str, Any]] = []
        scored_at = datetime.now(timezone.utc).isoformat()
        for prediction in prediction_records:
            match_id = str(prediction.get("match_id", ""))
            fixture = fixtures_by_id.get(match_id)
            if not fixture:
                continue

            status = self._text(fixture.get("status")).lower()
            if status not in COMPLETED_STATUSES:
                continue

            actual = self.actual_outcome(
                fixture.get("home_team", ""),
                fixture.get("away_team", ""),
                fixture.get("home_score"),
                fixture.get("away_score"),
            )
            predicted = self.predicted_outcome(
                prediction.get("prediction", ""),
                str(fixture.get("home_team", "")),
                str(fixture.get("away_team", "")),
            )
            if actual is None or predicted is None:
                continue

            scores.append(
                {
                    "match_id": match_id,
                    "match_date": fixture.get("match_date", ""),
                    "home_team": fixture.get("home_team", ""),
                    "away_team": fixture.get("away_team", ""),
                    "model": prediction.get("model", ""),
                    "prompt_version": prediction.get("prompt_version", ""),
                    "prediction": prediction.get("prediction", ""),
                    "actual_outcome": actual,
                    "points": 1 if predicted == actual else 0,
                    "status": "scored",
                    "scored_at": scored_at,
                }
            )

        if scores:
            return self.score_storage.save_scores(scores)
        return self.score_storage.load_scores()

    def leaderboard(self) -> pd.DataFrame:
        """Return model totals ordered by points and accuracy."""
        scores = self.score_storage.load_scores()
        if scores.empty:
            return pd.DataFrame(
                columns=["model", "predictions", "correct", "points", "accuracy"]
            )

        scores["points"] = pd.to_numeric(scores["points"], errors="coerce").fillna(0)
        rows: list[dict[str, Any]] = []
        for model, model_scores in scores.groupby("model", dropna=False):
            prediction_count = len(model_scores)
            points = int(model_scores["points"].sum())
            correct = int((model_scores["points"] > 0).sum())
            rows.append(
                {
                    "model": str(model),
                    "predictions": prediction_count,
                    "correct": correct,
                    "points": points,
                    "accuracy": round(correct / prediction_count, 4) if prediction_count else 0.0,
                }
            )

        return pd.DataFrame(rows).sort_values(
            ["points", "accuracy", "predictions", "model"],
            ascending=[False, False, False, True],
            ignore_index=True,
        )

    @staticmethod
    def actual_outcome(
        home_team: Any,
        away_team: Any,
        home_score: Any,
        away_score: Any,
    ) -> str | None:
        del home_team, away_team
        home = PredictionScorer._number(home_score)
        away = PredictionScorer._number(away_score)
        if home is None or away is None:
            return None
        if home > away:
            return "Home win"
        if away > home:
            return "Away win"
        return "Draw"

    @staticmethod
    def predicted_outcome(prediction: Any, home_team: str, away_team: str) -> str | None:
        text = PredictionScorer._text(prediction).lower()
        if not text:
            return None
        if text == "draw" or " draw " in f" {text} ":
            return "Draw"

        home = home_team.strip().lower()
        away = away_team.strip().lower()
        if home and (text == f"{home} win" or text.startswith(f"{home} ")):
            return "Home win"
        if away and (text == f"{away} win" or text.startswith(f"{away} ")):
            return "Away win"
        if text.startswith("home"):
            return "Home win"
        if text.startswith("away"):
            return "Away win"
        return None

    @staticmethod
    def _number(value: Any) -> int | None:
        text = PredictionScorer._text(value)
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _text(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text.lower() in {"", "nan", "none", "<na>"} else text
