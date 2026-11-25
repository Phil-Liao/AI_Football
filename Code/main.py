"""Central application orchestration for AI Football."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from football_data import FootballDataClient
from prediction import (
    get_prediction_models,
    predict_upcoming_for_models,
    start_daily_prediction_scheduler,
)
from scoring import PredictionScorer
from storage import FixtureStorage, PredictionStorage, ScoreStorage


BASE_DIR = Path(__file__).resolve().parent


class MainApplication:
    """Own the application's non-UI operations and services."""

    def __init__(
        self,
        base_dir: Path | None = None,
        football_client: FootballDataClient | None = None,
    ) -> None:
        self.base_dir = (base_dir or BASE_DIR).resolve()
        self.fixture_storage = FixtureStorage(self.base_dir / "data.csv")
        self.prediction_storage = PredictionStorage(self.base_dir / "predictions.csv")
        self.score_storage = ScoreStorage(self.base_dir / "scores.csv")
        self.football_client = football_client or FootballDataClient()
        self.scorer = PredictionScorer(
            self.prediction_storage,
            self.score_storage,
        )

    def start_daily_predictions(self) -> Any:
        """Start the host-local daily prediction schedule."""
        return start_daily_prediction_scheduler(
            self.fixture_storage,
            self.prediction_storage,
        )

    def refresh_fixtures(self) -> tuple[list[dict[str, Any]], int]:
        """Fetch yesterday/tomorrow fixtures and merge them into storage."""
        matches_by_date = self.football_client.get_yesterday_and_tomorrow_matches()
        refreshed_matches = [
            match
            for matches in matches_by_date.values()
            for match in matches
        ]
        self.fixture_storage.merge_matchday(refreshed_matches)
        return refreshed_matches, len(refreshed_matches)

    def load_dashboard(self) -> dict[str, Any]:
        """Load fixtures, predictions, scores, and model metadata for the UI."""
        fixtures = self.fixture_storage.load_matchday()
        self.scorer.score_completed_fixtures(fixtures)
        predictions = self.prediction_storage.load_predictions()
        scores = self.score_storage.load_scores()
        models = get_prediction_models()

        valid_predictions = predictions
        if not predictions.empty:
            valid_predictions = predictions.loc[
                predictions["prediction"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                & ~predictions["prediction"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.startswith("prediction unavailable")
            ]

        active_model_names = {
            model.model_id: model.name
            for model in models
        }

        model_summaries = []
        for model in models:
            prediction_count = 0
            if not valid_predictions.empty:
                prediction_count = int(
                    valid_predictions.loc[
                        valid_predictions["model"] == model.model_id,
                        "match_id",
                    ].nunique()
                )

            points = 0
            if not scores.empty:
                points = int(
                    pd.to_numeric(
                        scores.loc[
                            scores["model"] == model.model_id,
                            "points",
                        ],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )

            model_summaries.append(
                {
                    "name": model.name,
                    "model_id": model.model_id,
                    "prediction_count": prediction_count,
                    "points": points,
                }
            )

        return {
            "fixtures": fixtures,
            "predictions": predictions,
            "scores": scores,
            "leaderboard": self.scorer.leaderboard(),
            "active_model_names": active_model_names,
            "model_summaries": model_summaries,
        }

    def predict_missing(self, fixtures: pd.DataFrame) -> dict[str, dict[str, str]]:
        """Run every registered provider for fixtures missing a prediction."""
        return predict_upcoming_for_models(
            fixtures,
            self.prediction_storage,
            fixture_history=fixtures,
        )
