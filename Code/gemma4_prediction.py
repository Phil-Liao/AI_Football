"""Gemma implementation of the prediction model interface."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Iterable

import dotenv
import pandas as pd
from google import genai
from google.genai import errors


MODEL = "gemma-4-26b-a4b-it"
PROMPT_VERSION = "2"
MIN_REQUEST_INTERVAL_SECONDS = 2.0
MAX_503_RETRIES = 3
RETRY_BASE_SECONDS = 1.0

LOGGER = logging.getLogger(__name__)
_REQUEST_LOCK = threading.Lock()
_last_request_started = 0.0


def get_client() -> genai.Client:
    """Create a Gemini client from the GOOGLE_API_KEY environment variable."""
    dotenv.load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY is not configured")
    return genai.Client()


def _generate_content(client: genai.Client, prompt: str) -> Any:
    global _last_request_started
    with _REQUEST_LOCK:
        elapsed = time.monotonic() - _last_request_started
        if _last_request_started and elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        _last_request_started = time.monotonic()
        return client.models.generate_content(
            model=MODEL,
            contents=prompt,
        )


def _is_overloaded(error: errors.APIError) -> bool:
    return error.code == 503 and error.status == "UNAVAILABLE"


def predict_match(fixture: dict[str, Any]) -> str | None:
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

    historical_context = fixture.get("historical_context", [])
    if historical_context:
        prompt += (
            "\nPre-match context from earlier fixtures only. "
            "Do not infer the result from any information outside this context:\n"
            f"{json.dumps(historical_context, ensure_ascii=False)}\n"
        )

    try:
        client = get_client()
    except ValueError as error:
        LOGGER.warning("Gemma client unavailable: %s", error)
        return None

    for attempt in range(MAX_503_RETRIES + 1):
        try:
            response = _generate_content(client, prompt)
            return (response.text or "").strip() or None
        except errors.APIError as error:
            if not _is_overloaded(error):
                LOGGER.warning("Gemma prediction failed: %s", error)
                return None
            if attempt == MAX_503_RETRIES:
                LOGGER.warning(
                    "Gemma prediction still unavailable after %s attempts: %s",
                    attempt + 1,
                    error,
                )
                return None

            delay = RETRY_BASE_SECONDS * (2**attempt)
            LOGGER.warning(
                "Gemma is overloaded; retrying attempt %s of %s in %.1f seconds",
                attempt + 2,
                MAX_503_RETRIES + 1,
                delay,
            )
            time.sleep(delay)

    return None


def predict_fixtures(fixtures: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Predict a collection of fixtures, keyed by match_id."""
    predictions: dict[str, str] = {}
    for fixture in fixtures:
        match_id = str(fixture.get("match_id", ""))
        if not match_id:
            match_id = f"{fixture.get('home_team', '')}-{fixture.get('away_team', '')}"
        prediction = predict_match(fixture)
        if prediction and not prediction.lower().startswith("prediction unavailable"):
            predictions[match_id] = prediction
    return predictions


class GemmaPredictionModel:
    """Gemma 4 match prediction provider."""

    name = "gemma"
    model_id = MODEL
    prompt_version = PROMPT_VERSION

    def predict_match(self, fixture: dict[str, Any]) -> str | None:
        return predict_match(fixture)


def get_prediction_model() -> GemmaPredictionModel:
    """Create the configured Gemma prediction provider."""
    return GemmaPredictionModel()


def predict_upcoming(fixtures: pd.DataFrame, prediction_storage: Any | None = None) -> dict[str, str]:
    """Compatibility wrapper around the model-neutral prediction service."""
    from prediction import predict_upcoming as predict_upcoming_for_model

    return predict_upcoming_for_model(get_prediction_model(), fixtures, prediction_storage)
