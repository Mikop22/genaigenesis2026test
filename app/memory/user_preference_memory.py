"""Persistent user preference memory: load, save, apply feedback, sync visual focus."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.agents.derive_visual_focus import sync_visual_focus
from app.agents.extract_feedback import FeedbackSignals

LOG = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "created_at": _utcnow_iso(),
        "updated_at": _utcnow_iso(),
        "version": 1,
    }


def _features_from_dossier(listing: dict) -> list[str]:
    """Extract feature names from a listing dossier's visual_summary."""
    features: list[str] = []
    vs = listing.get("visual_summary", {})
    if vs:
        features.extend(vs.get("style", []))
        features.extend(vs.get("notable_features", []))
    return features


class UserPreferenceMemory:
    """Load/save ``UserPreferences`` and apply feedback-driven weight updates."""

    def __init__(self, storage_dir: str = "data/user_prefs"):
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, user_id: str) -> Path:
        safe_id = user_id.replace("/", "_").replace("\\", "_").replace("+", "_")
        return self._storage_dir / f"{safe_id}.json"

    def load(self, user_id: str) -> dict:
        """Load preferences for *user_id*. Returns defaults if none saved."""
        path = self._path_for(user_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                LOG.warning("Failed to load prefs for %s: %s", user_id, exc)
        return _default_preferences()

    def save(self, user_id: str, prefs: dict) -> None:
        """Persist preferences to a JSON file."""
        prefs["updated_at"] = _utcnow_iso()
        path = self._path_for(user_id)
        path.write_text(json.dumps(prefs, indent=2, default=str), encoding="utf-8")
        LOG.debug("Saved prefs for %s -> %s", user_id, path)

    def apply_feedback(
        self,
        prefs: dict,
        signals: FeedbackSignals,
        listing: dict,
        *,
        positive_step: float = 0.1,
        negative_step: float = 0.12,
        saturation_cap: float = 1.0,
        decay_factor: float = 0.02,
    ) -> None:
        """
        Adjust soft ranking_weights based on feedback signals from one interaction.

        - *positive_step*: how much to increase a weight on positive evidence.
        - *negative_step*: how much to decrease a weight on negative evidence
          (slightly higher because negative feedback tends to be more decisive).
        - *saturation_cap*: weights are clamped to ``[-cap, +cap]``.
        - *decay_factor*: all weights decay slightly toward 0 each update to
          prevent stale strong opinions from dominating forever.
        """
        weights = prefs.setdefault("ranking_weights", {})

        # 1. Decay all weights slightly toward zero
        for key in list(weights.keys()):
            w = weights[key]
            if isinstance(w, dict):
                w["value"] = w.get("value", 0.0) * (1.0 - decay_factor)
                w["updates"] = w.get("updates", 0) + 1
            else:
                weights[key] = w * (1.0 - decay_factor)

        # 2. Apply positive signals
        for feature in signals.positives:
            current = weights.get(feature, 0.0)
            if isinstance(current, dict):
                current["value"] = min(current.get("value", 0.0) + positive_step, saturation_cap)
                current["confidence"] = min(current.get("confidence", 0.0) + 0.05, 1.0)
                current["updates"] = current.get("updates", 0) + 1
            else:
                weights[feature] = min(current + positive_step, saturation_cap)

        # 3. Apply negative signals
        for feature in signals.negatives:
            current = weights.get(feature, 0.0)
            if isinstance(current, dict):
                current["value"] = max(current.get("value", 0.0) - negative_step, -saturation_cap)
                current["confidence"] = min(current.get("confidence", 0.0) + 0.05, 1.0)
                current["updates"] = current.get("updates", 0) + 1
            else:
                weights[feature] = max(current - negative_step, -saturation_cap)

        # 4. Strong positive (verdict == "yes") amplifies all positive features
        #    that appear in the listing's visual summary
        if signals.verdict == "yes":
            for feature in _features_from_dossier(listing):
                current = weights.get(feature, 0.0)
                if isinstance(current, dict):
                    current["value"] = min(
                        current.get("value", 0.0) + positive_step * 0.5,
                        saturation_cap,
                    )
                else:
                    weights[feature] = min(current + positive_step * 0.5, saturation_cap)

        # 5. Apply hard constraint updates from feedback
        hard = prefs.setdefault("hard_constraints", {})
        for update in signals.hard_constraint_updates:
            if update.get("type") == "add_no_go":
                no_go = hard.setdefault("no_go_features", [])
                val = update["value"]
                if val not in no_go:
                    no_go.append(val)
            elif update.get("type") == "set_min_beds":
                hard["min_beds"] = update["value"]

        # 6. Update liked/rejected examples
        url = listing.get("url", "")
        if signals.verdict == "yes":
            prefs.setdefault("liked_examples", []).append(
                {"url": url, "reason": signals.raw_text}
            )
        elif signals.verdict == "no":
            prefs.setdefault("rejected_examples", []).append(
                {"url": url, "reason": signals.raw_text}
            )

        prefs["updated_at"] = _utcnow_iso()

    def sync_visual_focus(self, prefs: dict) -> None:
        """Ensure high-weight features appear in ``visual_focus``."""
        sync_visual_focus(prefs)


def save_session(session: dict, storage_dir: str = "data/sessions") -> None:
    """Persist a ``SearchSession`` dict to disk for later resume."""
    path = Path(storage_dir)
    path.mkdir(parents=True, exist_ok=True)
    session_id = session.get("session_id", "unknown")
    file_path = path / f"{session_id}.json"
    file_path.write_text(json.dumps(session, indent=2, default=str), encoding="utf-8")
    LOG.debug("Session saved: %s", file_path)


def load_session(session_id: str, storage_dir: str = "data/sessions") -> Optional[dict]:
    """Load a previously saved ``SearchSession``."""
    file_path = Path(storage_dir) / f"{session_id}.json"
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOG.warning("Failed to load session %s: %s", session_id, exc)
        return None
