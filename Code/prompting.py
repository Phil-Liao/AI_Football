"""Shared prompt templates for prediction providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_PROMPT_CONFIG = json.loads(
    Path(__file__).with_name("prediction_prompts.json").read_text(encoding="utf-8")
)
_MATCH_OUTCOME_PROMPT = _PROMPT_CONFIG["match_outcome"]
PROMPT_VERSION = _MATCH_OUTCOME_PROMPT["version"]


def build_match_prediction_prompt(fixture: Mapping[str, Any]) -> str:
    """Build the shared match-outcome prompt for any prediction provider."""
    prompt = _MATCH_OUTCOME_PROMPT["template"].format(
        home_team=fixture.get("home_team", "Home team"),
        away_team=fixture.get("away_team", "Away team"),
        kickoff=fixture.get("match_date", "unknown date"),
    )

    historical_context = fixture.get("historical_context", [])
    if historical_context:
        prompt += _MATCH_OUTCOME_PROMPT["historical_context_template"].format(
            historical_context=json.dumps(historical_context, ensure_ascii=False)
        )

    return prompt
