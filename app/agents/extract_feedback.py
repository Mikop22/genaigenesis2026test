"""Extract structured feedback signals from user text."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

LOG = logging.getLogger(__name__)

# Simple keyword-based verdict detection (fallback when LLM is unavailable)
_YES_PATTERNS = re.compile(r"\b(yes|love it|perfect|contact|reach out|apply|interested)\b", re.I)
_NO_PATTERNS = re.compile(r"\b(no|pass|skip it|not for me|hate|terrible|awful|next)\b", re.I)
_MAYBE_PATTERNS = re.compile(r"\b(maybe|possibly|could work|not bad|decent|ok|okay|might work)\b", re.I)
_SKIP_PATTERNS = re.compile(r"\b(skip|come back|later|revisit|save for later)\b", re.I)


@dataclass
class FeedbackSignals:
    """Structured signals extracted from user feedback text."""

    verdict: str = "neutral"  # "yes" | "no" | "maybe" | "skip" | "neutral"
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    hard_constraint_updates: list[dict] = field(default_factory=list)
    raw_text: str = ""


def _keyword_verdict(text: str) -> str:
    """Determine verdict from keywords as a fallback."""
    if _YES_PATTERNS.search(text):
        return "yes"
    if _NO_PATTERNS.search(text):
        return "no"
    if _SKIP_PATTERNS.search(text):
        return "skip"
    if _MAYBE_PATTERNS.search(text):
        return "maybe"
    return "neutral"


# Common feature mappings for keyword extraction
_FEATURE_POSITIVE_MAP: dict[re.Pattern, str] = {
    re.compile(r"(love|like|great|nice|beautiful|amazing)\s+(the\s+)?kitchen", re.I): "updated_kitchen",
    re.compile(r"(bright|sunny|light|natural light|lots of light)", re.I): "natural_light",
    re.compile(r"(love|like|great|nice)\s+(the\s+)?bathroom", re.I): "modern_bathroom",
    re.compile(r"(outdoor|balcony|patio|terrace|garden|deck)", re.I): "outdoor_space",
    re.compile(r"(spacious|roomy|big|large|open|huge)\s+(room|space|apartment|living|bedroom)?", re.I): "spacious",
    re.compile(r"(high\s+floor|upper\s+floor|great\s+view|view)", re.I): "high_floor",
    re.compile(r"(storage|closet|walk-in)", re.I): "storage",
    re.compile(r"(quiet|peaceful|calm)", re.I): "quiet_location",
    re.compile(r"(hardwood|wood\s+floor)", re.I): "hardwood_floors",
    re.compile(r"(updated|renovated|modern|new)", re.I): "updated",
}

_FEATURE_NEGATIVE_MAP: dict[re.Pattern, str] = {
    re.compile(r"(old|outdated|dated|ugly)\s+(the\s+)?(bathroom|bath)", re.I): "dated_bathroom",
    re.compile(r"(dark|dim|no\s+light|not enough light)", re.I): "dark_rooms",
    re.compile(r"(noisy|loud|traffic|noise)", re.I): "noisy_street",
    re.compile(r"(small|tiny|cramped|tight)", re.I): "small_rooms",
    re.compile(r"(old|outdated|dated|ugly)\s+(the\s+)?kitchen", re.I): "dated_kitchen",
    re.compile(r"(ground\s+floor|first\s+floor|basement)", re.I): "ground_floor",
    re.compile(r"(no\s+(laundry|washer|dryer))", re.I): "no_laundry",
    re.compile(r"(no\s+(parking|garage))", re.I): "no_parking",
}

# Hard constraint triggers
_HARD_CONSTRAINT_PATTERNS: list[tuple[re.Pattern, dict]] = [
    (
        re.compile(r"(never|will not|won't|refuse).*(ground\s+floor|first\s+floor)", re.I),
        {"type": "add_no_go", "value": "ground floor"},
    ),
    (
        re.compile(r"(need|must|require)\s+(at\s+least\s+)?(\d+)\s+bed", re.I),
        {"type": "set_min_beds"},  # value filled dynamically
    ),
    (
        re.compile(r"(never|will not|won't).*(no\s+laundry|without\s+laundry)", re.I),
        {"type": "add_no_go", "value": "no laundry in unit"},
    ),
]


def _keyword_extract(text: str) -> FeedbackSignals:
    """Extract feedback signals using keyword patterns (no LLM)."""
    verdict = _keyword_verdict(text)

    positives: list[str] = []
    for pattern, feature in _FEATURE_POSITIVE_MAP.items():
        if pattern.search(text):
            positives.append(feature)

    negatives: list[str] = []
    for pattern, feature in _FEATURE_NEGATIVE_MAP.items():
        if pattern.search(text):
            negatives.append(feature)

    hard_updates: list[dict] = []
    for pattern, template in _HARD_CONSTRAINT_PATTERNS:
        m = pattern.search(text)
        if m:
            update = dict(template)
            if update.get("type") == "set_min_beds" and m.lastindex and m.lastindex >= 3:
                update["value"] = int(m.group(3))
            if "value" in update:
                hard_updates.append(update)

    return FeedbackSignals(
        verdict=verdict,
        positives=positives,
        negatives=negatives,
        hard_constraint_updates=hard_updates,
        raw_text=text,
    )


def _llm_extract(text: str) -> FeedbackSignals:
    """Extract feedback signals using the LLM."""
    system_prompt = (
        "You extract structured feedback from a user's comment about a real-estate listing.\n"
        "Return a JSON object with exactly these keys:\n"
        '  "verdict": one of "yes", "no", "maybe", "skip", "neutral"\n'
        '  "positives": list of feature names the user liked\n'
        '  "negatives": list of feature names the user disliked\n'
        '  "hard_constraint_updates": list of objects like {"type": "add_no_go", "value": "..."}\n'
        "Use lowercase feature names like: updated_kitchen, natural_light, dated_bathroom, "
        "dark_rooms, outdoor_space, high_floor, modern_bathroom, noisy_street, spacious, storage.\n"
        "Only return the JSON, no other text."
    )
    try:
        from app.config import GPT_OSS_BASE_URL, GPT_OSS_MODEL

        api_key = os.getenv("OPENAI_API_KEY", "not-set")
        import urllib.request

        payload = json.dumps(
            {
                "model": GPT_OSS_MODEL,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
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
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw_response = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        raw_response = raw_response.strip()
        if "```" in raw_response:
            start = raw_response.find("```")
            if "json" in raw_response[: start + 10]:
                start = raw_response.find("\n", start) + 1
            end = raw_response.find("```", start)
            raw_response = raw_response[start:end] if end > start else raw_response
        parsed = json.loads(raw_response)
        return FeedbackSignals(
            verdict=parsed.get("verdict", "neutral"),
            positives=parsed.get("positives", []),
            negatives=parsed.get("negatives", []),
            hard_constraint_updates=parsed.get("hard_constraint_updates", []),
            raw_text=text,
        )
    except Exception as exc:
        LOG.warning("LLM feedback extraction failed (%s), falling back to keywords", exc)
        return _keyword_extract(text)


def extract_feedback(text: str, *, use_llm: bool = True) -> FeedbackSignals:
    """
    Parse user feedback text into structured ``FeedbackSignals``.

    When ``use_llm`` is True (default), tries the LLM first and falls back to
    keyword-based extraction on failure.
    """
    if not text or not text.strip():
        return FeedbackSignals(raw_text=text or "")

    if use_llm:
        return _llm_extract(text)
    return _keyword_extract(text)
