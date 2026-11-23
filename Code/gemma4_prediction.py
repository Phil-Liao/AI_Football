"""Gemma-powered Premier League match predictions."""

from __future__ import annotations

import os
from typing import Any, Iterable

import dotenv
import pandas as pd
from google import genai

from storage import PredictionStorage


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


def predict_upcoming(
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
    predictions = predict_fixtures(upcoming.to_dict("records"))

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
                "model": MODEL,
                "prediction": prediction,
                "prompt_version": PROMPT_VERSION,
                "status": "stored",
            })
        prediction_storage.save_predictions(rows)

    return predictions
