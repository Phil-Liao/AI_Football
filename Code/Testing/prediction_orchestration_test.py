import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from google.genai import errors
from requests import ConnectionError, Response

import gemma4_prediction
import gemini_flash_lite_prediction
import nemotron_prediction
import openrouter_prediction
import prediction
from storage import FixtureStorage, PredictionStorage


class FakeModel:
    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self.prompt_version = "1"

    def predict_match(self, _fixture: dict[str, object]) -> str:
        return f"{self.name} prediction for {_fixture.get('home_team', '')}"


class PredictionOrchestrationTests(unittest.TestCase):
    @staticmethod
    def _overloaded_error() -> errors.ServerError:
        response = Response()
        response.status_code = 503
        response._content = (
            b'{"error":{"code":503,"message":"busy","status":"UNAVAILABLE"}}'
        )
        response._content_consumed = True
        return errors.ServerError(503, response)

    def test_prediction_models_are_discovered_from_provider_files(self) -> None:
        fake_models = {
            "alpha_prediction": FakeModel("Alpha", "alpha-id"),
            "beta_prediction": FakeModel("Beta", "beta-id"),
        }
        provider_modules = {
            module_name: unittest.mock.Mock(
                get_prediction_model=unittest.mock.Mock(return_value=model)
            )
            for module_name, model in fake_models.items()
        }

        with (
            patch.object(
                prediction.Path,
                "glob",
                return_value=[Path("beta_prediction.py"), Path("alpha_prediction.py")],
            ),
            patch.object(
                prediction,
                "import_module",
                side_effect=provider_modules.__getitem__,
            ),
        ):
            discovered = prediction.get_prediction_models()
            selected = prediction.get_prediction_model("beta-id")

        self.assertEqual([model.name for model in discovered], ["Alpha", "Beta"])
        self.assertIs(selected, fake_models["beta_prediction"])

    def test_predicts_only_fixtures_missing_for_each_registered_model(self) -> None:
        fixtures = pd.DataFrame(
            [
                {"match_id": "already-first", "home_team": "Home", "status": "complete"},
                {"match_id": "missing-completed", "home_team": "Home", "status": "complete"},
                {"match_id": "missing-upcoming", "home_team": "Home", "status": "scheduled"},
            ]
        )
        models = [FakeModel("first", "first-id"), FakeModel("second", "second-id")]

        with tempfile.TemporaryDirectory() as directory:
            storage = PredictionStorage(Path(directory) / "predictions.csv")
            storage.save_prediction(
                {
                    "match_id": "already-first",
                    "model": "first-id",
                    "prediction": "Home win",
                    "prompt_version": "1",
                }
            )
            with patch.object(prediction, "get_prediction_models", return_value=models):
                result = prediction.predict_upcoming_for_models(fixtures, storage)

            self.assertEqual(
                result,
                {
                    "first": {
                        "missing-completed": "first prediction for Home",
                        "missing-upcoming": "first prediction for Home",
                    },
                    "second": {
                        "already-first": "second prediction for Home",
                        "missing-completed": "second prediction for Home",
                        "missing-upcoming": "second prediction for Home",
                    },
                },
            )
            saved = storage.load_predictions()
            self.assertEqual(set(saved["model"]), {"first-id", "second-id"})
            self.assertEqual(
                set(saved["match_id"]),
                {"already-first", "missing-completed", "missing-upcoming"},
            )

    def test_model_input_hides_target_result_and_has_only_prior_context(self) -> None:
        class RecordingModel(FakeModel):
            def __init__(self) -> None:
                super().__init__("gemma", "gemma-id")
                self.inputs: list[dict[str, object]] = []

            def predict_match(self, fixture: dict[str, object]) -> str:
                self.inputs.append(fixture)
                return "Home win"

        fixtures = pd.DataFrame(
            [
                {
                    "match_id": "previous",
                    "match_date": "2099-01-01 12:00:00",
                    "home_team": "Earlier Home",
                    "away_team": "Earlier Away",
                    "home_score": "2",
                    "away_score": "0",
                    "status": "complete",
                    "raw_match": '{"winner_text":"Earlier Home won"}',
                },
                {
                    "match_id": "target",
                    "match_date": "2099-01-02 12:00:00",
                    "home_team": "Target Home",
                    "away_team": "Target Away",
                    "home_score": "999",
                    "away_score": "888",
                    "status": "complete",
                    "raw_match": '{"winner_text":"Target Home won"}',
                },
                {
                    "match_id": "same-time",
                    "match_date": "2099-01-02 12:00:00",
                    "home_team": "Same Time Home",
                    "away_team": "Same Time Away",
                    "home_score": "3",
                    "away_score": "0",
                    "status": "complete",
                },
                {
                    "match_id": "later",
                    "match_date": "2099-01-03 12:00:00",
                    "home_team": "Later Home",
                    "away_team": "Later Away",
                    "home_score": "0",
                    "away_score": "3",
                    "status": "complete",
                },
            ]
        )
        model = RecordingModel()

        with tempfile.TemporaryDirectory() as directory:
            storage = PredictionStorage(Path(directory) / "predictions.csv")
            storage.save_prediction(
                {
                    "match_id": "previous",
                    "model": "gemma-id",
                    "prediction": "Home win",
                    "prompt_version": "1",
                    "created_at": "2099-01-01T10:00:00+00:00",
                }
            )
            storage.save_prediction(
                {
                    "match_id": "previous",
                    "model": "gemma-id",
                    "prediction": "Draw",
                    "prompt_version": "2",
                    "created_at": "2099-01-02T13:00:00+00:00",
                }
            )
            prediction.predict_upcoming(model, fixtures, storage)

        target_input = next(
            item for item in model.inputs if item["home_team"] == "Target Home"
        )
        self.assertEqual(
            set(target_input),
            {"match_date", "home_team", "away_team", "historical_context"},
        )
        self.assertEqual(
            target_input["historical_context"],
            [
                {
                    "match_date": "2099-01-01 12:00:00",
                    "home_team": "Earlier Home",
                    "away_team": "Earlier Away",
                    "result": "2-0",
                    "predictions": [
                        {"model": "gemma-id", "prediction": "Home win"}
                    ],
                }
            ],
        )

    def test_gemma_prompt_ignores_target_scores_and_raw_match(self) -> None:
        client = unittest.mock.Mock()
        client.models.generate_content.return_value.text = "Draw"
        fixture = {
            "match_date": "2026-11-25 12:00:00",
            "home_team": "Target Home",
            "away_team": "Target Away",
            "home_score": "999",
            "away_score": "888",
            "raw_match": '{"winner_text":"TARGET_RESULT_TOKEN"}',
            "historical_context": [
                {
                    "home_team": "Earlier Home",
                    "away_team": "Earlier Away",
                    "result": "2-0",
                    "predictions": [
                        {"model": "gemma", "prediction": "Home win"}
                    ],
                }
            ],
        }

        with patch.object(gemma4_prediction, "get_client", return_value=client):
            result = gemma4_prediction.predict_match(fixture)

        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(result, "Draw")
        self.assertIn("Earlier Home", prompt)
        self.assertNotIn("TARGET_RESULT_TOKEN", prompt)
        self.assertNotIn("999", prompt)
        self.assertNotIn("888", prompt)

    def test_openrouter_ernie_uses_shared_prompt_and_configured_key(self) -> None:
        response = unittest.mock.Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Draw"}}]
        }
        fixture = {
            "match_date": "2026-11-25 12:00:00",
            "home_team": "Target Home",
            "away_team": "Target Away",
            "home_score": "999",
            "raw_match": '{"winner_text":"TARGET_RESULT_TOKEN"}',
        }

        with (
            patch.dict("os.environ", {"OPEN_ROUTER_KEY": "test-key"}),
            patch.object(openrouter_prediction.dotenv, "load_dotenv"),
            patch.object(openrouter_prediction.requests, "post", return_value=response) as post,
        ):
            result = openrouter_prediction.predict_match(fixture)

        self.assertEqual(result, "Draw")
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], openrouter_prediction.API_URL)
        self.assertEqual(
            post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key"
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["model"], openrouter_prediction.MODEL
        )
        self.assertEqual(
            post.call_args.kwargs["json"]["reasoning"], {"enabled": True}
        )
        prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("Target Home", prompt)
        self.assertNotIn("TARGET_RESULT_TOKEN", prompt)
        self.assertNotIn("999", prompt)

    def test_nemotron_preserves_reasoning_details_for_followup_request(self) -> None:
        reasoning_details = [{"type": "reasoning.test", "data": "opaque-value"}]
        first_response = unittest.mock.Mock()
        first_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Initial analysis",
                        "reasoning_details": reasoning_details,
                    }
                }
            ]
        }
        final_response = unittest.mock.Mock()
        final_response.json.return_value = {
            "choices": [{"message": {"content": "Away win"}}]
        }
        fixture = {
            "match_date": "2026-11-25 12:00:00",
            "home_team": "Target Home",
            "away_team": "Target Away",
            "home_score": "999",
            "raw_match": '{"winner_text":"TARGET_RESULT_TOKEN"}',
        }

        with (
            patch.dict("os.environ", {"OPEN_ROUTER_KEY": "test-key"}),
            patch.object(nemotron_prediction.dotenv, "load_dotenv"),
            patch.object(
                nemotron_prediction.requests,
                "post",
                side_effect=[first_response, final_response],
            ) as post,
        ):
            result = nemotron_prediction.predict_match(fixture)

        self.assertEqual(result, "Away win")
        self.assertEqual(post.call_count, 2)
        first_payload = post.call_args_list[0].kwargs["json"]
        second_payload = post.call_args_list[1].kwargs["json"]
        self.assertEqual(first_payload["model"], nemotron_prediction.MODEL)
        self.assertEqual(first_payload["reasoning"], {"enabled": True})
        self.assertEqual(second_payload["model"], nemotron_prediction.MODEL)
        self.assertEqual(second_payload["reasoning"], {"enabled": True})
        self.assertEqual(
            second_payload["messages"][1],
            {
                "role": "assistant",
                "content": "Initial analysis",
                "reasoning_details": reasoning_details,
            },
        )
        self.assertIs(
            second_payload["messages"][1]["reasoning_details"], reasoning_details
        )
        prompt = first_payload["messages"][0]["content"]
        self.assertIn("Target Home", prompt)
        self.assertNotIn("TARGET_RESULT_TOKEN", prompt)
        self.assertNotIn("999", prompt)

    def test_gemini_flash_lite_uses_second_key_timeout_and_shared_prompt(self) -> None:
        client = unittest.mock.Mock()
        client.models.generate_content.return_value.text = "Draw"
        fixture = {
            "match_date": "2026-11-25 12:00:00",
            "home_team": "Target Home",
            "away_team": "Target Away",
            "home_score": "999",
            "raw_match": '{"winner_text":"TARGET_RESULT_TOKEN"}',
        }

        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY2": "test-key"}),
            patch.object(gemini_flash_lite_prediction.dotenv, "load_dotenv"),
            patch.object(
                gemini_flash_lite_prediction.genai,
                "Client",
                return_value=client,
            ) as client_factory,
        ):
            result = gemini_flash_lite_prediction.predict_match(fixture)

        self.assertEqual(result, "Draw")
        client_factory.assert_called_once_with(
            api_key="test-key",
            http_options={
                "timeout": gemini_flash_lite_prediction.REQUEST_TIMEOUT_MS
            },
        )
        prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(
            client.models.generate_content.call_args.kwargs["model"],
            "gemini-3.1-flash-lite",
        )
        self.assertIn("Target Home", prompt)
        self.assertNotIn("TARGET_RESULT_TOKEN", prompt)
        self.assertNotIn("999", prompt)

    def test_gemini_flash_lite_retries_rate_limit_responses(self) -> None:
        response = Response()
        response.status_code = 429
        response._content = (
            b'{"error":{"code":429,"message":"busy","status":"RESOURCE_EXHAUSTED"}}'
        )
        response._content_consumed = True
        client = unittest.mock.Mock()
        client.models.generate_content.side_effect = [
            errors.ClientError(429, response),
            unittest.mock.Mock(text="Home win"),
        ]

        with (
            patch.object(
                gemini_flash_lite_prediction,
                "get_client",
                return_value=client,
            ),
            patch.object(
                gemini_flash_lite_prediction,
                "_last_request_started",
                0.0,
            ),
            patch.object(
                gemini_flash_lite_prediction,
                "MIN_REQUEST_INTERVAL_SECONDS",
                0.0,
            ),
            patch.object(gemini_flash_lite_prediction.time, "sleep") as sleep,
        ):
            result = gemini_flash_lite_prediction.predict_match(
                {"home_team": "Home", "away_team": "Away"}
            )

        self.assertEqual(result, "Home win")
        self.assertEqual(client.models.generate_content.call_count, 2)
        sleep.assert_called_once_with(
            gemini_flash_lite_prediction.RETRY_BASE_SECONDS
        )

    def test_gemma_retries_503_and_spaces_api_requests(self) -> None:
        client = unittest.mock.Mock()
        client.models.generate_content.side_effect = [
            self._overloaded_error(),
            unittest.mock.Mock(text="Home win"),
        ]

        with (
            patch.object(gemma4_prediction, "get_client", return_value=client),
            patch.object(gemma4_prediction, "_last_request_started", 0.0),
            patch.object(gemma4_prediction.time, "monotonic", return_value=100.0),
            patch.object(gemma4_prediction.time, "sleep") as sleep,
        ):
            result = gemma4_prediction.predict_match(
                {"home_team": "Home", "away_team": "Away"}
            )

        self.assertEqual(result, "Home win")
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [gemma4_prediction.RETRY_BASE_SECONDS, gemma4_prediction.MIN_REQUEST_INTERVAL_SECONDS],
        )

    def test_exhausted_503_is_ignored_and_not_saved(self) -> None:
        class OverloadedModel(FakeModel):
            def predict_match(self, fixture: dict[str, object]) -> str:
                return f"Prediction unavailable: 503 UNAVAILABLE for {fixture.get('home_team', '')}"

        fixtures = pd.DataFrame(
            [{"match_id": "busy", "home_team": "Home", "status": "complete"}]
        )

        with tempfile.TemporaryDirectory() as directory:
            storage = PredictionStorage(Path(directory) / "predictions.csv")
            with patch.object(
                prediction,
                "get_prediction_models",
                return_value=[OverloadedModel("gemma", "gemma-id")],
            ):
                result = prediction.predict_upcoming_for_models(fixtures, storage)

            self.assertEqual(result, {"gemma": {}})
            self.assertTrue(storage.load_predictions().empty)

    def test_persistent_503_returns_no_prediction(self) -> None:
        client = unittest.mock.Mock()
        client.models.generate_content.side_effect = [
            self._overloaded_error()
            for _ in range(gemma4_prediction.MAX_503_RETRIES + 1)
        ]

        with (
            patch.object(gemma4_prediction, "get_client", return_value=client),
            patch.object(gemma4_prediction, "_last_request_started", 0.0),
            patch.object(gemma4_prediction.time, "monotonic", return_value=100.0),
            patch.object(gemma4_prediction.time, "sleep"),
            self.assertLogs("gemma4_prediction", level="WARNING"),
        ):
            result = gemma4_prediction.predict_match(
                {"home_team": "Home", "away_team": "Away"}
            )

        self.assertIsNone(result)
        self.assertEqual(
            client.models.generate_content.call_count,
            gemma4_prediction.MAX_503_RETRIES + 1,
        )

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
            result, {"gemma": {"tomorrow": "gemma prediction for Home"}}
        )

    def test_daily_run_returns_empty_when_api_is_unavailable(self) -> None:
        fake_client = unittest.mock.Mock()
        fake_client.get_matches.side_effect = ConnectionError("network unavailable")

        with tempfile.TemporaryDirectory() as directory:
            fixture_storage = FixtureStorage(Path(directory) / "fixtures.csv")
            prediction_storage = PredictionStorage(Path(directory) / "predictions.csv")
            result = prediction.run_daily_predictions(
                fixture_storage,
                prediction_storage,
                fake_client,
                today=date(2026, 11, 24),
            )

        self.assertEqual(result, {})

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
