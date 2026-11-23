"""Model-neutral prediction orchestration."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Iterable, Protocol

import pandas as pd

from storage import PredictionStorage


class PredictionModel(Protocol):
    """Interface implemented by every match prediction provider."""

    name: str
    model_id: str
    prompt_version: str

    def predict_match(self, fixture: dict[str, Any]) -> str:
        """Return a prediction for one fixture."""


_MODEL_MODULES = {
    "gemma": "gemma4_prediction",
}


def get_prediction_model(name: str = "gemma") -> PredictionModel:
    """Resolve a prediction provider by its public name."""
    try:
        module_name = _MODEL_MODULES[name.lower()]
    except KeyError as error:
        available = ", ".join(sorted(_MODEL_MODULES))
        raise ValueError(f"Unknown prediction model '{name}'. Available: {available}") from error

    module = import_module(module_name)
    factory = getattr(module, "get_prediction_model", None)
    if factory is None:
        raise ImportError(f"Provider '{name}' does not expose get_prediction_model")
    return factory()


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
