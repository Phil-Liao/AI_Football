import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import prediction
from storage import FixtureStorage, PredictionStorage


class FakeModel:
    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self.prompt_version = "1"

    def predict_match(self, _fixture: dict[str, object]) -> str:
        return f"{self.name} prediction for {_fixture.get('match_id', '')}"


class PredictionOrchestrationTests(unittest.TestCase):
    def test_upcoming_predictions_call_every_registered_model(self) -> None:
        fixtures = pd.DataFrame(
            [
                {"match_id": "tomorrow", "status": "scheduled"},
                {"match_id": "finished", "status": "complete"},
            ]
        )
        models = [FakeModel("first", "first-id"), FakeModel("second", "second-id")]

        with tempfile.TemporaryDirectory() as directory:
            storage = PredictionStorage(Path(directory) / "predictions.csv")
            with patch.object(prediction, "get_prediction_models", return_value=models):
                result = prediction.predict_upcoming_for_models(fixtures, storage)

            self.assertEqual(
                result,
                {
                    "first": {"tomorrow": "first prediction for tomorrow"},
                    "second": {"tomorrow": "second prediction for tomorrow"},
                },
            )
            saved = storage.load_predictions()
            self.assertEqual(set(saved["model"]), {"first-id", "second-id"})
            self.assertEqual(set(saved["match_id"]), {"tomorrow"})

    def test_daily_run_fetches_and_predicts_only_tomorrow(self) -> None:
        fixed_today = date(2026, 11, 24)
        fake_client = unittest.mock.Mock()
        fake_client.get_matches.return_value = [
            {
                "match_id": "tomorrow",
                "match_date": "2026-11-25 12:00:00",
                "home_team": "Home",
                "away_team": "Away",
                "status": "scheduled",
            },
            {
                "match_id": "later",
                "match_date": "2026-11-26 12:00:00",
                "home_team": "Later Home",
                "away_team": "Later Away",
                "status": "scheduled",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            fixture_storage = FixtureStorage(Path(directory) / "fixtures.csv")
            prediction_storage = PredictionStorage(Path(directory) / "predictions.csv")
            models = [FakeModel("gemma", "gemma-4")]
            with patch.object(prediction, "get_prediction_models", return_value=models):
                result = prediction.run_daily_predictions(
                    fixture_storage,
                    prediction_storage,
                    fake_client,
                    today=fixed_today,
                )

        fake_client.get_matches.assert_called_once_with(date(2026, 11, 25))
        self.assertEqual(
            result, {"gemma": {"tomorrow": "gemma prediction for tomorrow"}}
        )

    @patch("prediction.BackgroundScheduler")
    def test_scheduler_runs_on_startup_and_repeats_daily(self, scheduler_factory) -> None:
        scheduler = scheduler_factory.return_value

        prediction.start_daily_prediction_scheduler()

        job = scheduler.add_job.call_args
        trigger = job.kwargs["trigger"]
        self.assertEqual(str(trigger.fields[5]), "6")
        self.assertEqual(str(trigger.fields[6]), "0")
        self.assertIsNotNone(job.kwargs["next_run_time"])
        scheduler.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
