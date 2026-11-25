"""Model-neutral prediction orchestration."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from importlib import import_module
from typing import Any, Iterable, Protocol, TypeGuard

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from football_data import FootballDataClient
from scoring import PredictionScorer
from storage import FixtureStorage, PredictionStorage, ScoreStorage


LOGGER = logging.getLogger(__name__)
DAILY_PREDICTION_HOUR = 6


class PredictionModel(Protocol):
    """Interface implemented by every match prediction provider."""

    name: str
    model_id: str
    prompt_version: str

    def predict_match(self, fixture: dict[str, Any]) -> str | None:
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
    fixture_records = list(fixtures)
    predictions: dict[str, str] = {}
    for fixture in fixture_records:
        match_id = _fixture_key(fixture)
        model_input = _build_model_input(fixture, fixture_records, [])
        prediction = model.predict_match(model_input)
        if _is_available_prediction(prediction):
            predictions[match_id] = prediction
    return predictions


def predict_upcoming_for_models(
    fixtures: pd.DataFrame,
    prediction_storage: PredictionStorage | None = None,
    model_names: Iterable[str] | None = None,
    *,
    fixture_history: pd.DataFrame | None = None,
) -> dict[str, dict[str, str]]:
    """Predict every fixture missing a saved prediction from each provider."""
    predictions: dict[str, dict[str, str]] = {}
    for model in get_prediction_models(model_names):
        predictions[model.name] = predict_upcoming(
            model,
            fixtures,
            prediction_storage,
            fixture_history=fixture_history,
        )
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
    score_storage = ScoreStorage()
    football_client = football_client or FootballDataClient()

    PredictionScorer(prediction_storage, score_storage).score_completed_fixtures(
        fixture_storage.load_matchday()
    )

    tomorrow = (today or date.today()) + timedelta(days=1)

    matches = football_client.get_matches(tomorrow)
    if not matches:
        LOGGER.info("No fixtures found for %s; skipping daily predictions", tomorrow)
        return {}

    fixture_storage.merge_matchday(matches)
    fixtures = fixture_storage.load_matchday()
    fixture_dates = pd.to_datetime(fixtures["match_date"], errors="coerce").dt.date
    tomorrow_fixtures = fixtures.loc[fixture_dates == tomorrow]
    return predict_upcoming_for_models(
        tomorrow_fixtures,
        prediction_storage,
        fixture_history=fixtures,
    )


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
    *,
    fixture_history: pd.DataFrame | None = None,
    prediction_history: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Predict fixtures missing this model's result, regardless of match status."""
    if fixtures.empty:
        return {}

    history_records = (
        fixture_history.to_dict("records")
        if fixture_history is not None
        else fixtures.to_dict("records")
    )
    if prediction_history is None:
        prediction_history = (
            prediction_storage.load_predictions()
            if prediction_storage is not None
            else pd.DataFrame()
        )
    prediction_records = prediction_history.to_dict("records")

    already_predicted = {
        _fixture_key(row)
        for row in prediction_records
        if _text(row.get("model")) == model.model_id
        and _is_available_prediction(row.get("prediction"))
    }
    candidates = [
        row
        for row in fixtures.to_dict("records")
        if _fixture_key(row) not in already_predicted
    ]
    candidates.sort(key=_kickoff_sort_key)

    predictions: dict[str, str] = {}
    rows_to_save: list[dict[str, Any]] = []
    for fixture in candidates:
        match_id = _fixture_key(fixture)
        model_input = _build_model_input(
            fixture, history_records, prediction_records
        )
        prediction = model.predict_match(model_input)
        if not _is_available_prediction(prediction):
            continue
        predictions[match_id] = prediction

        created_at = datetime.now(timezone.utc).isoformat()
        prediction_record = {
            "match_id": match_id,
            "match_date": fixture.get("match_date", ""),
            "home_team": fixture.get("home_team", ""),
            "away_team": fixture.get("away_team", ""),
            "model": model.model_id,
            "prediction": prediction,
            "prompt_version": model.prompt_version,
            "status": "stored",
            "created_at": created_at,
        }
        prediction_records.append(prediction_record)
        rows_to_save.append(prediction_record)

    if prediction_storage is not None and rows_to_save:
        prediction_storage.save_predictions(rows_to_save)

    return predictions


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def _fixture_key(fixture: dict[str, Any]) -> str:
    match_id = _text(fixture.get("match_id"))
    if match_id:
        return match_id
    return f"{_text(fixture.get('home_team'))}-{_text(fixture.get('away_team'))}"


def _is_available_prediction(value: Any) -> TypeGuard[str]:
    prediction = _text(value)
    return (
        isinstance(value, str)
        and bool(prediction)
        and not prediction.lower().startswith("prediction unavailable")
    )


def _parse_utc(value: Any) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(timestamp) else timestamp


def _kickoff_sort_key(fixture: dict[str, Any]) -> int:
    kickoff = _parse_utc(fixture.get("match_date"))
    return kickoff.value if kickoff is not None else 2**63 - 1


def _build_model_input(
    fixture: dict[str, Any],
    fixture_history: list[dict[str, Any]],
    prediction_history: list[dict[str, Any]],
) -> dict[str, Any]:
    kickoff = _parse_utc(fixture.get("match_date"))
    match_id = _fixture_key(fixture)
    prior_fixtures: dict[str, tuple[pd.Timestamp, dict[str, Any]]] = {}
    if kickoff is not None:
        for past_fixture in fixture_history:
            past_match_id = _fixture_key(past_fixture)
            past_kickoff = _parse_utc(past_fixture.get("match_date"))
            if (
                past_match_id != match_id
                and past_kickoff is not None
                and past_kickoff < kickoff
            ):
                prior_fixtures[past_match_id] = (past_kickoff, past_fixture)

    prior_predictions: dict[str, dict[str, tuple[pd.Timestamp, dict[str, str]]]] = (
        defaultdict(dict)
    )
    for row in prediction_history:
        past_match_id = _fixture_key(row)
        if past_match_id not in prior_fixtures or kickoff is None:
            continue
        created_at = _parse_utc(row.get("created_at"))
        model_id = _text(row.get("model"))
        prediction = _text(row.get("prediction"))
        if (
            created_at is None
            or created_at >= kickoff
            or not model_id
            or not _is_available_prediction(prediction)
        ):
            continue

        previous = prior_predictions[past_match_id].get(model_id)
        if previous is None or created_at > previous[0]:
            prior_predictions[past_match_id][model_id] = (
                created_at,
                {"model": model_id, "prediction": prediction},
            )

    context: list[dict[str, Any]] = []
    for past_match_id, (past_kickoff, past_fixture) in sorted(
        prior_fixtures.items(), key=lambda item: item[1][0]
    ):
        scores = (
            _text(past_fixture.get("home_score")),
            _text(past_fixture.get("away_score")),
        )
        status = _text(past_fixture.get("status")).lower()
        has_result = (
            status in {"complete", "completed", "finished"}
            and all(scores)
        )
        prior_match_predictions = [
            prediction
            for _, prediction in sorted(
                prior_predictions[past_match_id].values(),
                key=lambda item: item[1]["model"],
            )
        ]
        if not has_result and not prior_match_predictions:
            continue

        history_item: dict[str, Any] = {
            "match_date": _text(past_fixture.get("match_date")),
            "home_team": _text(past_fixture.get("home_team")),
            "away_team": _text(past_fixture.get("away_team")),
        }
        if has_result:
            history_item["result"] = f"{scores[0]}-{scores[1]}"
        if prior_match_predictions:
            history_item["predictions"] = prior_match_predictions
        context.append(history_item)

    return {
        "match_date": _text(fixture.get("match_date")),
        "home_team": _text(fixture.get("home_team")),
        "away_team": _text(fixture.get("away_team")),
        "historical_context": context,
    }
