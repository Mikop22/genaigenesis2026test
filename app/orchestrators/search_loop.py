"""Iterative multi-page Zillow search orchestrator.

Manages the search→analyze→rank→present→feedback loop described in the plan.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from app.agents.derive_visual_focus import sync_visual_focus
from app.agents.extract_feedback import FeedbackSignals, extract_feedback
from app.agents.rank_listings import rank_listings
from app.agents.summarize_listing_images import summarize_listing_images
from app.memory.user_preference_memory import (
    UserPreferenceMemory,
    save_session,
)

LOG = logging.getLogger(__name__)

# When the unshown queue drops below this, fetch the next Zillow page
REFILL_THRESHOLD = 3


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SearchSession helpers
# ---------------------------------------------------------------------------

def new_session(
    user_id: str,
    criteria: dict,
    session_id: Optional[str] = None,
) -> dict:
    """Create a fresh ``SearchSession`` dict."""
    return {
        "session_id": session_id or f"sess_{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "created_at": _utcnow_iso(),
        "criteria": dict(criteria),
        "next_page_cursor": 1,
        "listings": {},
        "contact_log": [],
        "feedback_history": [],
        "user_done": False,
    }


def _set_listing_state(session: dict, url: str, state: str, **extra) -> None:
    entry = session["listings"].setdefault(url, {"discovered_at": _utcnow_iso()})
    entry["state"] = state
    entry.update(extra)


def _state_listings(session: dict, state: str) -> list[str]:
    """Return URLs in *session* that are currently in *state*."""
    return [
        url for url, info in session["listings"].items() if info.get("state") == state
    ]


def _unshown_listings(session: dict) -> list[str]:
    """URLs that have been ranked but not yet shown."""
    return _state_listings(session, "ranked")


def _top_ranked_unshown(session: dict) -> Optional[str]:
    """URL of the highest-scored unshown listing, or None."""
    candidates = []
    for url, info in session["listings"].items():
        if info.get("state") == "ranked":
            candidates.append((url, info.get("score", 0.0)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Dossier builder
# ---------------------------------------------------------------------------

def build_dossier(url: str, user_prefs: dict, *, headless: bool = False) -> dict:
    """Scrape a listing detail page and produce a full dossier with visual summary."""
    from data.zillow.detail_scraper import scrape_listing_detail

    detail = scrape_listing_detail(url, headless=headless)
    visual_summary = summarize_listing_images(
        detail,
        visual_focus=user_prefs.get("visual_focus", []),
    )
    description = detail.get("description", "")
    # Truncate long descriptions to a summary length
    description_summary = description[:500] if description else ""
    return {
        **detail,
        "visual_summary": visual_summary,
        "description_summary": description_summary,
    }


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def run_search_loop(
    session: dict,
    user_prefs: dict,
    *,
    present_to_user: Callable[[dict], Optional[str]],
    on_contact: Optional[Callable[[str, dict], dict]] = None,
    headless: bool = False,
    memory: Optional[UserPreferenceMemory] = None,
    session_storage_dir: str = "data/sessions",
    max_pages: int = 10,
) -> dict:
    """
    Run the iterative search-rank-present-feedback loop.

    Parameters
    ----------
    session : dict
        A ``SearchSession`` dict (from ``new_session``).
    user_prefs : dict
        Current ``UserPreferences``.
    present_to_user : callable
        ``(listing_dossier) -> feedback_text | None``.  Blocks until the user
        responds.  Return ``None`` to signal the user is done.
    on_contact : callable, optional
        ``(url, lead) -> ContactResult``.  Called when the user says "yes".
        If ``None``, the contact step is skipped.
    headless : bool
        Whether to run the browser headless.
    memory : UserPreferenceMemory, optional
        If supplied, preferences and session are persisted after each feedback.
    session_storage_dir : str
        Directory for session persistence (used when *memory* is set).
    max_pages : int
        Safety limit on how many Zillow pages to fetch.
    """
    from data.zillow.paginated_search import paginated_search

    while not session.get("user_done"):
        # 1. Refill the work queue if running low
        if len(_unshown_listings(session)) < REFILL_THRESHOLD:
            page = session["next_page_cursor"]
            if page > max_pages:
                LOG.info("Reached max page limit (%d), stopping.", max_pages)
                break
            LOG.info("Fetching search page %d …", page)
            raw = paginated_search(session["criteria"], page=page, headless=headless)
            for link in raw.get("listing_links", []):
                if link not in session["listings"]:
                    _set_listing_state(session, link, "discovered")
            session["next_page_cursor"] = page + 1

        # 2. Analyse un-analysed listings
        for url in _state_listings(session, "discovered"):
            LOG.info("Building dossier for %s …", url)
            try:
                dossier = build_dossier(url, user_prefs, headless=headless)
                _set_listing_state(session, url, "analyzed", dossier=dossier)
            except Exception as exc:
                LOG.warning("Dossier build failed for %s: %s", url, exc)
                _set_listing_state(session, url, "rejected", error=str(exc))

        # 3. Rank analysed listings
        analyzed_urls = _state_listings(session, "analyzed")
        if analyzed_urls:
            dossiers = [session["listings"][u]["dossier"] for u in analyzed_urls]
            ranked = rank_listings(dossiers, user_prefs)
            for url, score in ranked:
                _set_listing_state(session, url, "ranked", score=score)
            # Mark any that were excluded by hard constraints as rejected
            ranked_urls = {u for u, _ in ranked}
            for url in analyzed_urls:
                if url not in ranked_urls:
                    _set_listing_state(session, url, "rejected", reason="hard_constraint")

        # 4. Pick best unshown listing
        best_url = _top_ranked_unshown(session)
        if best_url is None:
            LOG.info("No unshown listings available, fetching more.")
            continue

        listing_info = session["listings"][best_url]
        dossier = listing_info.get("dossier", {})
        _set_listing_state(session, best_url, "shown")

        # 5. Present to user and collect feedback
        feedback_text = present_to_user(dossier)

        if feedback_text is None:
            session["user_done"] = True
            break

        # 6. Process feedback
        signals: FeedbackSignals = extract_feedback(feedback_text)

        # Record feedback in session
        listing_info["feedback"] = {
            "raw_text": signals.raw_text,
            "signals": {
                "verdict": signals.verdict,
                "positives": signals.positives,
                "negatives": signals.negatives,
            },
        }
        session["feedback_history"].append(
            {
                "listing_url": best_url,
                "feedback_text": signals.raw_text,
                "verdict": signals.verdict,
            }
        )

        # 7. Update preference weights
        if memory:
            memory.apply_feedback(user_prefs, signals, dossier)
            memory.sync_visual_focus(user_prefs)
            memory.save(session["user_id"], user_prefs)
        else:
            _apply_feedback_inline(user_prefs, signals, dossier)
            sync_visual_focus(user_prefs)

        # 8. Handle verdict
        if signals.verdict == "yes":
            _set_listing_state(session, best_url, "liked")
            if on_contact:
                LOG.info("User said yes — triggering contact flow for %s", best_url)
                _set_listing_state(session, best_url, "contact_requested")
                try:
                    result = on_contact(best_url, session)
                    session["contact_log"].append(
                        {
                            "url": best_url,
                            "attempted_at": _utcnow_iso(),
                            "result": result,
                        }
                    )
                    _set_listing_state(session, best_url, "contacted")
                except Exception as exc:
                    LOG.warning("Contact flow failed for %s: %s", best_url, exc)
            # Continue searching after contact
        elif signals.verdict in ("no", "neutral") or signals.negatives:
            _set_listing_state(session, best_url, "rejected")
        elif signals.verdict == "maybe":
            _set_listing_state(session, best_url, "liked")
        elif signals.verdict == "skip":
            _set_listing_state(session, best_url, "deferred")

        # 9. Persist session
        if memory:
            save_session(session, storage_dir=session_storage_dir)

    return session


# ---------------------------------------------------------------------------
# Inline weight update (when no persistent memory is configured)
# ---------------------------------------------------------------------------

def _apply_feedback_inline(
    prefs: dict,
    signals: FeedbackSignals,
    listing: dict,
    *,
    positive_step: float = 0.1,
    negative_step: float = 0.12,
    saturation_cap: float = 1.0,
    decay_factor: float = 0.02,
) -> None:
    """Lightweight weight update without persistence (Phase 1 MVP)."""
    weights = prefs.setdefault("ranking_weights", {})

    # Decay
    for key in list(weights.keys()):
        w = weights[key]
        if isinstance(w, (int, float)):
            weights[key] = w * (1.0 - decay_factor)

    # Positives
    for feature in signals.positives:
        current = weights.get(feature, 0.0)
        if isinstance(current, (int, float)):
            weights[feature] = min(current + positive_step, saturation_cap)

    # Negatives
    for feature in signals.negatives:
        current = weights.get(feature, 0.0)
        if isinstance(current, (int, float)):
            weights[feature] = max(current - negative_step, -saturation_cap)

    # "yes" amplification
    if signals.verdict == "yes":
        vs = listing.get("visual_summary", {})
        for feature in (vs.get("style", []) + vs.get("notable_features", [])):
            current = weights.get(feature, 0.0)
            if isinstance(current, (int, float)):
                weights[feature] = min(current + positive_step * 0.5, saturation_cap)

    prefs["updated_at"] = _utcnow_iso()
