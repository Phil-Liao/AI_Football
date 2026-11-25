"""Gemini Flash-Lite implementation of the prediction model interface."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import dotenv
import pandas as pd
from google import genai
from google.genai import errors

from prompting import PROMPT_VERSION, build_match_prediction_prompt


MODEL = "gemini-3.1-flash-lite"
MIN_REQUEST_INTERVAL_SECONDS = 2.0
REQUEST_TIMEOUT_MS = 60_000
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0

LOGGER = logging.getLogger(__name__)
_REQUEST_LOCK = threading.Lock()
_last_request_started = 0.0


def get_client() -> genai.Client:
    """Create a Gemini client from the GOOGLE_API_KEY2 environment variable."""
    dotenv.load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY2")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY2 is not configured")
    return genai.Client(
        api_key=api_key,
        http_options={"timeout": REQUEST_TIMEOUT_MS},
    )


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


def _is_retryable(error: errors.APIError) -> bool:
    return error.code in {429, 503}


def predict_match(fixture: dict[str, Any]) -> str | None:
    """Return a short predicted outcome for one Premier League fixture."""
    prompt = build_match_prediction_prompt(fixture)

    try:
        client = get_client()
    except ValueError as error:
        LOGGER.warning("Gemini Flash-Lite client unavailable: %s", error)
        return None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = _generate_content(client, prompt)
            return (response.text or "").strip() or None
        except errors.APIError as error:
            if not _is_retryable(error):
                LOGGER.warning("Gemini Flash-Lite prediction failed: %s", error)
                return None
            if attempt == MAX_RETRIES:
                LOGGER.warning(
                    "Gemini Flash-Lite prediction still unavailable after %s attempts: %s",
                    attempt + 1,
                    error,
                )
                return None

            delay = RETRY_BASE_SECONDS * (2**attempt)
            LOGGER.warning(
                "Gemini Flash-Lite is rate-limited or overloaded; "
                "retrying attempt %s of %s in %.1f seconds",
                attempt + 2,
                MAX_RETRIES + 1,
                delay,
            )
            time.sleep(delay)

    return None


class GeminiFlashLitePredictionModel:
    """Gemini 3.1 Flash-Lite match prediction provider."""

    name = "gemini-3.1-flash-lite"
    model_id = MODEL
    prompt_version = PROMPT_VERSION

    def predict_match(self, fixture: dict[str, Any]) -> str | None:
        return predict_match(fixture)


def get_prediction_model() -> GeminiFlashLitePredictionModel:
    """Create the configured Gemini Flash-Lite prediction provider."""
    return GeminiFlashLitePredictionModel()


def predict_upcoming(
    fixtures: pd.DataFrame, prediction_storage: Any | None = None
) -> dict[str, str]:
    """Compatibility wrapper around the model-neutral prediction service."""
    from prediction import predict_upcoming as predict_upcoming_for_model

    return predict_upcoming_for_model(
        get_prediction_model(), fixtures, prediction_storage
    )
