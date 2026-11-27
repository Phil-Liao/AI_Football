"""Nemotron implementation of the prediction model interface via OpenRouter."""

from __future__ import annotations

import logging
import os
from typing import Any

import dotenv
import pandas as pd
import requests

from prompting import PROMPT_VERSION, build_match_prediction_prompt


MODEL = "nvidia/nemotron-3.5-lightning:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60

LOGGER = logging.getLogger(__name__)


def get_api_key() -> str:
    """Load the OpenRouter API key from the environment or local .env file."""
    dotenv.load_dotenv()
    api_key = os.getenv("OPEN_ROUTER_KEY2")
    if not api_key:
        raise ValueError("OPEN_ROUTER_KEY is not configured")
    return api_key


def _request_completion(api_key: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "reasoning": {"enabled": True},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]


def predict_match(fixture: dict[str, Any]) -> str | None:
    """Return a short predicted outcome for one Premier League fixture."""
    try:
        api_key = get_api_key()
    except ValueError as error:
        LOGGER.warning("OpenRouter client unavailable: %s", error)
        return None

    prompt = build_match_prediction_prompt(fixture)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        first_message = _request_completion(api_key, messages)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": first_message.get("content"),
                    "reasoning_details": first_message.get("reasoning_details"),
                },
                {"role": "user", "content": "Are you sure? Think carefully."},
            ]
        )
        final_message = _request_completion(api_key, messages)
        return (final_message.get("content") or "").strip() or None
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
        LOGGER.warning("Nemotron prediction failed: %s", error)
        return None


class NemotronPredictionModel:
    """Nemotron 3.5 Lightning match prediction provider."""

    name = "nemotron-3.5-lightning"
    model_id = MODEL
    prompt_version = PROMPT_VERSION

    def predict_match(self, fixture: dict[str, Any]) -> str | None:
        return predict_match(fixture)


def get_prediction_model() -> NemotronPredictionModel:
    """Create the configured Nemotron prediction provider."""
    return NemotronPredictionModel()


def predict_upcoming(
    fixtures: pd.DataFrame, prediction_storage: Any | None = None
) -> dict[str, str]:
    """Compatibility wrapper around the model-neutral prediction service."""
    from prediction import predict_upcoming as predict_upcoming_for_model

    return predict_upcoming_for_model(
        get_prediction_model(), fixtures, prediction_storage
    )