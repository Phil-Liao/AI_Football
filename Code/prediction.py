"""Model-neutral prediction orchestration."""

from __future__ import annotations

import logging
from importlib import import_module
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Protocol

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from football_data import FootballDataClient
from storage import FixtureStorage, PredictionStorage


LOGGER = logging.getLogger(__name__)
DAILY_PREDICTION_HOUR = 6


class PredictionModel(Protocol):
    """Interface implemented by every match prediction provider."""

    name: str
    model_id: str
    prompt_version: str

    def predict_match(self, fixture: dict[str, Any]) -> str:
        """Return a prediction for one fixture."""


MODEL_MODULES = {
    "gemma": "gemma4_prediction",
}


def get_prediction_model(name: str = "gemma") -> PredictionModel:
    """Resolve a prediction provider by its public name."""
    try:
        module_name = MODEL_MODULES[name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_MODULES))
        raise ValueError(f"Unknown prediction model '{name}'. Available: {available}") from error

    module = import_module(module_name)
    factory = getattr(module, "get_prediction_model", None)
    if factory is None:
        raise ImportError(f"Provider '{name}' does not expose get_prediction_model")
    return factory()


def get_prediction_models(names: Iterable[str] | None = None) -> list[PredictionModel]:
    """Resolve all registered providers, or a requested subset."""
    model_names = MODEL_MODULES if names is None else names
    return [get_prediction_model(name) for name in model_names]


def predict_fixtures(
    model: PredictionModel,
    fixtures: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Predict fixtures with any compatible model, keyed by match_id."""
    predictions: dict[str, str] = {}
    for fixture in fixtures:
        match_id = str(fixture.get("match_id", ""))
        if not match_id:
            match_id = f"{fixture.get('home_team', '')}-{fixture.get('away_team', '')}"
        predictions[match_id] = model.predict_match(fixture)
    return predictions


def predict_upcoming_for_models(
    fixtures: pd.DataFrame,
    prediction_storage: PredictionStorage | None = None,
    model_names: Iterable[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Predict incomplete fixtures with every selected provider."""
    predictions: dict[str, dict[str, str]] = {}
    for model in get_prediction_models(model_names):
        predictions[model.name] = predict_upcoming(model, fixtures, prediction_storage)
    return predictions


def run_daily_predictions(
    fixture_storage: FixtureStorage | None = None,
    prediction_storage: PredictionStorage | None = None,
    football_client: FootballDataClient | None = None,
    *,
    today: date | None = None,
) -> dict[str, dict[str, str]]:
    """Fetch and predict tomorrow's fixtures with every registered provider."""
    fixture_storage = fixture_storage or FixtureStorage()
    prediction_storage = prediction_storage or PredictionStorage()
    football_client = football_client or FootballDataClient()
    tomorrow = (today or date.today()) + timedelta(days=1)

    matches = football_client.get_matches(tomorrow)
    if not matches:
        LOGGER.info("No fixtures found for %s; skipping daily predictions", tomorrow)
        return {}

    fixture_storage.merge_matchday(matches)
    fixtures = fixture_storage.load_matchday()
    fixture_dates = pd.to_datetime(fixtures["match_date"], errors="coerce").dt.date
    tomorrow_fixtures = fixtures.loc[fixture_dates == tomorrow]
    return predict_upcoming_for_models(tomorrow_fixtures, prediction_storage)


def start_daily_prediction_scheduler(
    fixture_storage: FixtureStorage | None = None,
    prediction_storage: PredictionStorage | None = None,
) -> BackgroundScheduler:
    """Schedule tomorrow's fixture predictions daily at 06:00 host-local time."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_daily_predictions,
        trigger=CronTrigger(hour=DAILY_PREDICTION_HOUR, minute=0),
        args=(fixture_storage, prediction_storage),
        id="daily_tomorrow_predictions",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        next_run_time=datetime.now().astimezone(),
    )
    scheduler.start()
    return scheduler


def predict_upcoming(
    model: PredictionModel,
    fixtures: pd.DataFrame,
    prediction_storage: PredictionStorage | None = None,
) -> dict[str, str]:
    """Predict incomplete fixtures and optionally persist the results."""
    if fixtures.empty:
        return {}

    upcoming = fixtures[
        ~fixtures["status"].fillna("").astype(str).str.lower().isin(
            {"complete", "completed", "finished"}
        )
    ]
    predictions = predict_fixtures(model, upcoming.to_dict("records"))

    if prediction_storage is not None:
        rows: list[dict[str, Any]] = []
        for fixture in upcoming.to_dict("records"):
            match_id = str(fixture.get("match_id", ""))
            prediction = predictions.get(match_id, "")
            if not prediction or prediction.startswith("Prediction unavailable"):
                continue
            rows.append({
                "match_id": match_id,
                "match_date": fixture.get("match_date", ""),
                "home_team": fixture.get("home_team", ""),
                "away_team": fixture.get("away_team", ""),
                "model": model.model_id,
                "prediction": prediction,
                "prompt_version": model.prompt_version,
                "status": "stored",
            })
        prediction_storage.save_predictions(rows)

    return predictions
