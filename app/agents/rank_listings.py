"""Rank listing dossiers against user preferences using hard constraints and soft weights."""

from __future__ import annotations

import logging
import re
from typing import Optional

LOG = logging.getLogger(__name__)


def _numeric(value) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if value is None or value == "":
        return None
    try:
        return float(re.sub(r"[^0-9.]", "", str(value)))
    except (ValueError, TypeError):
        return None


def violates_hard_constraints(dossier: dict, hard_constraints: dict) -> bool:
    """
    Return True if the listing violates any hard constraint.

    Hard constraints are binary disqualifiers — a violating listing is excluded
    entirely rather than scored lower.
    """
    # Price ceiling
    max_price = _numeric(hard_constraints.get("max_price"))
    listing_price = _numeric(dossier.get("price"))
    if max_price is not None and listing_price is not None:
        if listing_price > max_price:
            return True

    # Minimum bedrooms
    min_beds = _numeric(hard_constraints.get("min_beds"))
    listing_beds = _numeric(dossier.get("beds"))
    if min_beds is not None and listing_beds is not None:
        if listing_beds < min_beds:
            return True

    # No-go features (check description and visual summary)
    no_go = hard_constraints.get("no_go_features", [])
    if no_go:
        text_pool = _text_pool(dossier)
        for feature in no_go:
            if feature.lower() in text_pool:
                return True

    return False


def _text_pool(dossier: dict) -> str:
    """Combine all textual information from a dossier into one lowercase string."""
    parts = [
        dossier.get("description", "") or dossier.get("description_summary", ""),
        dossier.get("address", ""),
    ]
    vs = dossier.get("visual_summary", {})
    if vs:
        parts.extend(vs.get("style", []))
        parts.extend(vs.get("notable_features", []))
        parts.extend(vs.get("possible_issues", []))
    facts = dossier.get("facts", {})
    if facts:
        parts.extend(str(v) for v in facts.values())
    return " ".join(parts).lower()


def _features_from_dossier(dossier: dict) -> list[str]:
    """Extract feature names from a dossier's visual_summary."""
    features: list[str] = []
    vs = dossier.get("visual_summary", {})
    if vs:
        features.extend(vs.get("style", []))
        features.extend(vs.get("notable_features", []))
    return features


def score_dossier(dossier: dict, ranking_weights: dict) -> float:
    """
    Score a single dossier against soft ranking weights.

    Each weight key is a feature name with a float value in [-1.0, 1.0].
    Positive weights boost score when the feature is present; negative weights
    penalise.
    """
    text_pool = _text_pool(dossier)
    score = 0.0
    for feature, weight in ranking_weights.items():
        # Extract numeric value if weight uses confidence tracking format
        w = weight if isinstance(weight, (int, float)) else weight.get("value", 0.0)
        if feature.lower() in text_pool:
            score += w
    return score


def rank_listings(
    dossiers: list[dict],
    user_prefs: dict,
) -> list[tuple[str, float]]:
    """
    Score and sort dossiers.

    Returns a list of ``(url, score)`` tuples in descending score order.
    Listings that violate hard constraints are excluded.
    """
    hard = user_prefs.get("hard_constraints", {})
    weights = user_prefs.get("ranking_weights", {})

    scored: list[tuple[str, float]] = []
    for dossier in dossiers:
        if violates_hard_constraints(dossier, hard):
            LOG.debug("Excluded %s (hard constraint violation)", dossier.get("url"))
            continue
        s = score_dossier(dossier, weights)
        scored.append((dossier["url"], s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
