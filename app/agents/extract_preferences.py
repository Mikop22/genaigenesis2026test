"""Extract initial user preferences from an onboarding transcript or chat."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

LOG = logging.getLogger(__name__)


def _default_preferences() -> dict:
    """Return a blank UserPreferences dict."""
    return {
        "hard_constraints": {
            "max_price": None,
            "min_beds": None,
            "locations": [],
            "no_go_features": [],
        },
        "ranking_weights": {},
        "stated_preferences": [],
        "visual_focus": [],
        "liked_examples": [],
        "rejected_examples": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "version": 1,
    }


def _parse_prefs_json(text: str) -> dict:
    """Parse LLM response into preferences dict."""
    text = (text or "").strip()
    if "```" in text:
        start = text.find("```")
        if "json" in text[: start + 10]:
            start = text.find("\n", start) + 1
        end = text.find("```", start)
        text = text[start:end] if end > start else text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


_SYSTEM_PROMPT = """\
You extract structured rental/purchase preferences from a user's conversation transcript.

Return a JSON object with exactly these keys:
{
  "max_price": <number or null>,
  "min_beds": <number or null>,
  "locations": ["<location1>", ...],
  "no_go_features": ["<feature1>", ...],
  "ranking_weights": {"<feature>": <float -1.0 to 1.0>, ...},
  "stated_preferences": ["<preference1>", ...],
  "visual_focus": ["<instruction1>", ...]
}

For ranking_weights use lowercase feature names like: natural_light, updated_kitchen,
dated_bathroom, dark_rooms, outdoor_space, high_floor, modern_bathroom, noisy_street.

For visual_focus, write instructions like "check if the kitchen is updated" or
"look for natural light in every room".

Only return the JSON, no other text.\
"""


def extract_preferences(transcript: str) -> dict:
    """
    Parse a user transcript into initial ``UserPreferences``.

    Tries the LLM first; falls back to a blank preference set on failure.
    """
    prefs = _default_preferences()

    if not transcript or not transcript.strip():
        return prefs

    try:
        from app.config import GPT_OSS_BASE_URL, GPT_OSS_MODEL

        api_key = os.getenv("OPENAI_API_KEY", "not-set")
        import urllib.request

        payload = json.dumps(
            {
                "model": GPT_OSS_MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url=f"{GPT_OSS_BASE_URL.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed = _parse_prefs_json(text)

        # Merge parsed fields into prefs
        hard = prefs["hard_constraints"]
        if parsed.get("max_price") is not None:
            hard["max_price"] = parsed["max_price"]
        if parsed.get("min_beds") is not None:
            hard["min_beds"] = parsed["min_beds"]
        if parsed.get("locations"):
            hard["locations"] = parsed["locations"]
        if parsed.get("no_go_features"):
            hard["no_go_features"] = parsed["no_go_features"]
        if parsed.get("ranking_weights"):
            prefs["ranking_weights"] = parsed["ranking_weights"]
        if parsed.get("stated_preferences"):
            prefs["stated_preferences"] = parsed["stated_preferences"]
        if parsed.get("visual_focus"):
            prefs["visual_focus"] = parsed["visual_focus"]

    except Exception as exc:
        LOG.warning("Preference extraction failed (%s), returning defaults", exc)

    return prefs
