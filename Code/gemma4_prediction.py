"""Gemma implementation of the prediction model interface."""

from __future__ import annotations

import os
from typing import Any, Iterable

import dotenv
import pandas as pd
from google import genai


MODEL = "gemma-4-26b-a4b-it"
PROMPT_VERSION = "1"


def get_client() -> genai.Client:
    """Create a Gemini client from the GOOGLE_API_KEY environment variable."""
    dotenv.load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not configured")
    return genai.Client()


def predict_match(fixture: dict[str, Any]) -> str:
    """Return a short predicted outcome for one Premier League fixture."""
    home_team = fixture.get("home_team", "Home team")
    away_team = fixture.get("away_team", "Away team")
    kickoff = fixture.get("match_date", "unknown date")

    prompt = (
        "Predict the result of this English Premier League match. "
        "Use only the teams and available match context. "
        "Reply with exactly one outcome, choose one of: "
        f"{home_team} win, Draw, {away_team} win. "
        "Do not add an explanation.\n\n"
        f"Kickoff: {kickoff}\n"
        f"Home team: {home_team}\n"
        f"Away team: {away_team}\n"
    )

    raw_match = fixture.get("raw_match")
    if raw_match:
        prompt += f"Available match data: {raw_match}\n"

    try:
        response = get_client().models.generate_content(
            model=MODEL,
            contents=prompt,
        )
    except Exception as error:
        return f"Prediction unavailable: {error}"

    return (response.text or "Prediction unavailable").strip()


def predict_fixtures(fixtures: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Predict a collection of fixtures, keyed by match_id."""
    predictions: dict[str, str] = {}
    for fixture in fixtures:
        match_id = str(fixture.get("match_id", ""))
        if not match_id:
            match_id = f"{fixture.get('home_team', '')}-{fixture.get('away_team', '')}"
        predictions[match_id] = predict_match(fixture)
    return predictions


class GemmaPredictionModel:
    """Gemma 4 match prediction provider."""

    name = "gemma"
    model_id = MODEL
    prompt_version = PROMPT_VERSION

    def predict_match(self, fixture: dict[str, Any]) -> str:
        return predict_match(fixture)


def get_prediction_model() -> GemmaPredictionModel:
    """Create the configured Gemma prediction provider."""
    return GemmaPredictionModel()


def predict_upcoming(fixtures: pd.DataFrame, prediction_storage: Any | None = None) -> dict[str, str]:
    """Compatibility wrapper around the model-neutral prediction service."""
    from prediction import predict_upcoming as predict_upcoming_for_model

    return predict_upcoming_for_model(get_prediction_model(), fixtures, prediction_storage)
