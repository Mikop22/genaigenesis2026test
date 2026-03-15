"""Auto-derive visual_focus prompts from ranking weights and stated preferences."""

from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)

# Template phrases for converting weight keys to visual_focus instructions
_POSITIVE_TEMPLATE = "check if listing has: {feature}"
_NEGATIVE_TEMPLATE = "flag if listing shows: {feature}"

# Thresholds for auto-adding features to visual_focus
POSITIVE_THRESHOLD = 0.6
NEGATIVE_THRESHOLD = -0.5


def derive_visual_focus(user_prefs: dict) -> list[str]:
    """
    Generate ``visual_focus`` prompts from ``ranking_weights`` and
    ``stated_preferences``.

    Returns a new list (does not modify ``user_prefs`` in-place).
    """
    focus: list[str] = list(user_prefs.get("visual_focus", []))
    existing_lower = {f.lower() for f in focus}
    weights = user_prefs.get("ranking_weights", {})

    for feature, weight in weights.items():
        # Handle both simple float and confidence-tracking dict formats
        w = weight if isinstance(weight, (int, float)) else weight.get("value", 0.0)

        if w >= POSITIVE_THRESHOLD:
            prompt = _POSITIVE_TEMPLATE.format(feature=feature)
            if prompt.lower() not in existing_lower:
                focus.append(prompt)
                existing_lower.add(prompt.lower())
        elif w <= NEGATIVE_THRESHOLD:
            prompt = _NEGATIVE_TEMPLATE.format(feature=feature)
            if prompt.lower() not in existing_lower:
                focus.append(prompt)
                existing_lower.add(prompt.lower())

    return focus


def sync_visual_focus(user_prefs: dict) -> None:
    """
    Update ``user_prefs["visual_focus"]`` in-place so that high-weight features
    appear as attention prompts for the image summarizer.
    """
    user_prefs["visual_focus"] = derive_visual_focus(user_prefs)
