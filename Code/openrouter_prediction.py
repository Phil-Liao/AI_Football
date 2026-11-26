"""ERNIE implementation of the prediction model interface via OpenRouter."""

from __future__ import annotations

import logging
import os
from typing import Any

import dotenv
import pandas as pd
import requests

from prompting import PROMPT_VERSION, build_match_prediction_prompt


MODEL = "baidu/ernie-5.5-flash-preview:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60

LOGGER = logging.getLogger(__name__)


def get_api_key() -> str:
    """Load the OpenRouter API key from the environment or local .env file."""
    dotenv.load_dotenv()
    api_key = os.getenv("OPEN_ROUTER_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_KEY is not configured")
    return api_key


def predict_match(fixture: dict[str, Any]) -> str | None:
    """Return a short predicted outcome for one Premier League fixture."""
    try:
        api_key = get_api_key()
    except ValueError as error:
        LOGGER.warning("OpenRouter client unavailable: %s", error)
        return None

    prompt = build_match_prediction_prompt(fixture)
    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "reasoning": {"enabled": True},
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        return (message.get("content") or "").strip() or None
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
        LOGGER.warning("OpenRouter prediction failed: %s", error)
        return None


class OpenRouterErniePredictionModel:
    """ERNIE 5.5 Flash Preview match prediction provider."""

    name = "openrouter-ernie"
    model_id = MODEL
    prompt_version = PROMPT_VERSION

    def predict_match(self, fixture: dict[str, Any]) -> str | None:
        return predict_match(fixture)


def get_prediction_model() -> OpenRouterErniePredictionModel:
    """Create the configured OpenRouter ERNIE prediction provider."""
    return OpenRouterErniePredictionModel()


def predict_upcoming(
    fixtures: pd.DataFrame, prediction_storage: Any | None = None
) -> dict[str, str]:
    """Compatibility wrapper around the model-neutral prediction service."""
    from prediction import predict_upcoming as predict_upcoming_for_model

    return predict_upcoming_for_model(
        get_prediction_model(), fixtures, prediction_storage
    )