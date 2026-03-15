"""Tests for build_search_url pagination, rank_listings, extract_feedback, derive_visual_focus, and user_preference_memory."""

import tempfile

import pytest

# ---------------------------------------------------------------------------
# 1. build_search_url with pagination
# ---------------------------------------------------------------------------

from data.zillow.scraper import build_search_url


class TestBuildSearchUrl:
    def test_default_page(self):
        url = build_search_url({"location": "Brooklyn NY", "intent": "rent"})
        assert "for_rent" in url
        assert "Brooklyn" in url
        # No pagination segment for page 1
        assert "_p/" not in url

    def test_page_2(self):
        url = build_search_url({"location": "Brooklyn NY", "intent": "rent"}, page=2)
        assert "2_p/" in url
        # Pagination segment should come before the query string
        assert url.index("2_p/") < url.index("?")

    def test_page_5(self):
        url = build_search_url({"location": "Brooklyn NY", "intent": "rent"}, page=5)
        assert "5_p/" in url
        assert url.index("5_p/") < url.index("?")

    def test_buy_maps_to_sale(self):
        url = build_search_url({"location": "NYC", "intent": "buy"})
        assert "for_sale" in url
        assert "for_buy" not in url


# ---------------------------------------------------------------------------
# 2. rank_listings
# ---------------------------------------------------------------------------

from app.agents.rank_listings import (
    rank_listings,
    score_dossier,
    violates_hard_constraints,
)


class TestRankListings:
    def _dossier(self, url="https://example.com/a", price="3000", beds="2", **kw):
        return {"url": url, "price": price, "beds": beds, "description": "", "visual_summary": {}, "facts": {}, **kw}

    def test_hard_constraint_price(self):
        d = self._dossier(price="4000")
        assert violates_hard_constraints(d, {"max_price": 3500})

    def test_hard_constraint_beds(self):
        d = self._dossier(beds="1")
        assert violates_hard_constraints(d, {"min_beds": 2})

    def test_no_violation(self):
        d = self._dossier(price="3000", beds="2")
        assert not violates_hard_constraints(d, {"max_price": 3500, "min_beds": 2})

    def test_no_go_feature(self):
        d = self._dossier(description="ground floor apartment")
        assert violates_hard_constraints(d, {"no_go_features": ["ground floor"]})

    def test_rank_order(self):
        d1 = self._dossier(url="http://a", description="bright natural light")
        d2 = self._dossier(url="http://b", description="dark rooms")
        prefs = {
            "hard_constraints": {},
            "ranking_weights": {"natural_light": 0.9, "dark_rooms": -0.8},
        }
        ranked = rank_listings([d1, d2], prefs)
        # d1 should score higher
        urls = [u for u, _ in ranked]
        assert urls[0] == "http://a"

    def test_score_with_confidence_dict(self):
        d = self._dossier(description="natural light everywhere")
        score = score_dossier(d, {"natural_light": {"value": 0.9, "confidence": 0.5, "updates": 3}})
        assert score == pytest.approx(0.9)

    def test_hard_constraint_excludes(self):
        d = self._dossier(price="5000")
        prefs = {"hard_constraints": {"max_price": 3500}, "ranking_weights": {}}
        ranked = rank_listings([d], prefs)
        assert len(ranked) == 0


# ---------------------------------------------------------------------------
# 3. extract_feedback (keyword mode — no LLM)
# ---------------------------------------------------------------------------

from app.agents.extract_feedback import FeedbackSignals, extract_feedback


class TestExtractFeedback:
    def test_yes_verdict(self):
        signals = extract_feedback("yes", use_llm=False)
        assert signals.verdict == "yes"

    def test_no_verdict(self):
        signals = extract_feedback("no", use_llm=False)
        assert signals.verdict == "no"

    def test_maybe_verdict(self):
        signals = extract_feedback("maybe", use_llm=False)
        assert signals.verdict == "maybe"

    def test_skip_verdict(self):
        signals = extract_feedback("skip", use_llm=False)
        assert signals.verdict == "skip"

    def test_positive_kitchen(self):
        signals = extract_feedback("I love the kitchen!", use_llm=False)
        assert "updated_kitchen" in signals.positives

    def test_negative_dark(self):
        signals = extract_feedback("too dark for me", use_llm=False)
        assert "dark_rooms" in signals.negatives

    def test_negative_bathroom(self):
        signals = extract_feedback("the bathroom is dated and ugly", use_llm=False)
        assert "dated_bathroom" in signals.negatives

    def test_empty_input(self):
        signals = extract_feedback("", use_llm=False)
        assert signals.verdict == "neutral"
        assert signals.positives == []
        assert signals.negatives == []

    def test_raw_text_preserved(self):
        signals = extract_feedback("love natural light", use_llm=False)
        assert signals.raw_text == "love natural light"

    def test_skip_it_produces_skip(self):
        signals = extract_feedback("skip it", use_llm=False)
        assert signals.verdict == "skip"

    def test_not_interested_produces_no(self):
        signals = extract_feedback("not interested", use_llm=False)
        assert signals.verdict != "yes"

    def test_light_alone_does_not_match_natural_light(self):
        signals = extract_feedback("the light fixture is broken", use_llm=False)
        assert "natural_light" not in signals.positives

    def test_new_york_does_not_match_updated(self):
        signals = extract_feedback("apartment in New York", use_llm=False)
        assert "updated" not in signals.positives


# ---------------------------------------------------------------------------
# 4. derive_visual_focus
# ---------------------------------------------------------------------------

from app.agents.derive_visual_focus import derive_visual_focus, sync_visual_focus


class TestDeriveVisualFocus:
    def test_high_positive_added(self):
        prefs = {"ranking_weights": {"natural_light": 0.9}, "visual_focus": []}
        focus = derive_visual_focus(prefs)
        assert any("natural_light" in f for f in focus)

    def test_high_negative_added(self):
        prefs = {"ranking_weights": {"dated_bathroom": -0.7}, "visual_focus": []}
        focus = derive_visual_focus(prefs)
        assert any("dated_bathroom" in f for f in focus)

    def test_low_weight_not_added(self):
        prefs = {"ranking_weights": {"storage": 0.3}, "visual_focus": []}
        focus = derive_visual_focus(prefs)
        assert not any("storage" in f for f in focus)

    def test_existing_preserved(self):
        prefs = {
            "ranking_weights": {"natural_light": 0.9},
            "visual_focus": ["check kitchen"],
        }
        focus = derive_visual_focus(prefs)
        assert "check kitchen" in focus

    def test_no_duplicates(self):
        prefs = {
            "ranking_weights": {"natural_light": 0.9},
            "visual_focus": ["check if listing has: natural_light"],
        }
        focus = derive_visual_focus(prefs)
        count = sum(1 for f in focus if "natural_light" in f)
        assert count == 1

    def test_sync_modifies_in_place(self):
        prefs = {"ranking_weights": {"natural_light": 0.9}, "visual_focus": []}
        sync_visual_focus(prefs)
        assert len(prefs["visual_focus"]) > 0

    def test_confidence_dict_weights(self):
        prefs = {
            "ranking_weights": {"natural_light": {"value": 0.9, "confidence": 0.8, "updates": 5}},
            "visual_focus": [],
        }
        focus = derive_visual_focus(prefs)
        assert any("natural_light" in f for f in focus)


# ---------------------------------------------------------------------------
# 5. UserPreferenceMemory
# ---------------------------------------------------------------------------

from app.memory.user_preference_memory import (
    UserPreferenceMemory,
    load_session,
    save_session,
)


class TestUserPreferenceMemory:
    def test_load_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            mem = UserPreferenceMemory(storage_dir=d)
            prefs = mem.load("test_user")
            assert prefs["hard_constraints"]["max_price"] is None
            assert prefs["ranking_weights"] == {}

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            mem = UserPreferenceMemory(storage_dir=d)
            prefs = mem.load("u1")
            prefs["ranking_weights"]["natural_light"] = 0.5
            mem.save("u1", prefs)
            loaded = mem.load("u1")
            assert loaded["ranking_weights"]["natural_light"] == 0.5

    def test_apply_feedback_positives(self):
        with tempfile.TemporaryDirectory() as d:
            mem = UserPreferenceMemory(storage_dir=d)
            prefs = mem.load("u1")
            signals = FeedbackSignals(
                verdict="maybe",
                positives=["updated_kitchen"],
                negatives=["dated_bathroom"],
                raw_text="kitchen nice, bathroom old",
            )
            listing = {"url": "http://example.com", "visual_summary": {}}
            mem.apply_feedback(prefs, signals, listing)
            assert prefs["ranking_weights"]["updated_kitchen"] > 0
            assert prefs["ranking_weights"]["dated_bathroom"] < 0

    def test_apply_feedback_yes_amplifies(self):
        with tempfile.TemporaryDirectory() as d:
            mem = UserPreferenceMemory(storage_dir=d)
            prefs = mem.load("u1")
            signals = FeedbackSignals(
                verdict="yes",
                positives=["natural_light"],
                raw_text="yes",
            )
            listing = {
                "url": "http://example.com",
                "visual_summary": {
                    "style": ["natural_light", "hardwood_floors"],
                    "notable_features": [],
                },
            }
            mem.apply_feedback(prefs, signals, listing)
            # natural_light gets +0.1 (positive) + 0.05 (amplification) = 0.15
            assert prefs["ranking_weights"]["natural_light"] > 0.1
            # hardwood_floors gets 0.05 from amplification
            assert prefs["ranking_weights"]["hardwood_floors"] > 0

    def test_apply_feedback_hard_constraint_update(self):
        with tempfile.TemporaryDirectory() as d:
            mem = UserPreferenceMemory(storage_dir=d)
            prefs = mem.load("u1")
            signals = FeedbackSignals(
                verdict="no",
                hard_constraint_updates=[{"type": "add_no_go", "value": "ground floor"}],
                raw_text="never ground floor",
            )
            listing = {"url": "http://example.com", "visual_summary": {}}
            mem.apply_feedback(prefs, signals, listing)
            assert "ground floor" in prefs["hard_constraints"]["no_go_features"]

    def test_apply_feedback_records_examples(self):
        with tempfile.TemporaryDirectory() as d:
            mem = UserPreferenceMemory(storage_dir=d)
            prefs = mem.load("u1")
            signals = FeedbackSignals(verdict="yes", raw_text="love it")
            listing = {"url": "http://example.com/liked", "visual_summary": {}}
            mem.apply_feedback(prefs, signals, listing)
            assert len(prefs["liked_examples"]) == 1
            assert prefs["liked_examples"][0]["url"] == "http://example.com/liked"

    def test_decay(self):
        with tempfile.TemporaryDirectory() as d:
            mem = UserPreferenceMemory(storage_dir=d)
            prefs = mem.load("u1")
            prefs["ranking_weights"]["storage"] = 0.5
            signals = FeedbackSignals(verdict="neutral", raw_text="meh")
            listing = {"url": "http://example.com", "visual_summary": {}}
            mem.apply_feedback(prefs, signals, listing)
            # 0.5 * (1 - 0.02) = 0.49
            assert prefs["ranking_weights"]["storage"] == pytest.approx(0.49)

    def test_saturation_cap(self):
        with tempfile.TemporaryDirectory() as d:
            mem = UserPreferenceMemory(storage_dir=d)
            prefs = mem.load("u1")
            prefs["ranking_weights"]["natural_light"] = 0.99
            signals = FeedbackSignals(
                verdict="yes",
                positives=["natural_light"],
                raw_text="love light",
            )
            listing = {"url": "http://example.com", "visual_summary": {"style": ["natural_light"], "notable_features": []}}
            mem.apply_feedback(prefs, signals, listing)
            assert prefs["ranking_weights"]["natural_light"] <= 1.0


class TestSessionPersistence:
    def test_save_and_load_session(self):
        with tempfile.TemporaryDirectory() as d:
            session = {
                "session_id": "test_sess",
                "user_id": "u1",
                "listings": {},
                "feedback_history": [],
            }
            save_session(session, storage_dir=d)
            loaded = load_session("test_sess", storage_dir=d)
            assert loaded is not None
            assert loaded["session_id"] == "test_sess"

    def test_load_missing_session(self):
        with tempfile.TemporaryDirectory() as d:
            assert load_session("nonexistent", storage_dir=d) is None

    def test_path_traversal_sanitised(self):
        with tempfile.TemporaryDirectory() as d:
            session = {
                "session_id": "../../etc/evil",
                "user_id": "u1",
                "listings": {},
                "feedback_history": [],
            }
            save_session(session, storage_dir=d)
            # The file should be saved safely inside the storage dir
            loaded = load_session("../../etc/evil", storage_dir=d)
            assert loaded is not None
            assert loaded["session_id"] == "../../etc/evil"


# ---------------------------------------------------------------------------
# 6. SearchSession helpers (orchestrator)
# ---------------------------------------------------------------------------

from app.orchestrators.search_loop import (
    new_session,
    _set_listing_state,
    _state_listings,
    _unshown_listings,
    _top_ranked_unshown,
    _apply_feedback_inline,
)


class TestSearchSessionHelpers:
    def test_new_session_fields(self):
        s = new_session("user1", {"location": "NYC", "intent": "rent"})
        assert s["user_id"] == "user1"
        assert s["next_page_cursor"] == 1
        assert s["listings"] == {}
        assert s["user_done"] is False

    def test_set_and_get_state(self):
        s = new_session("u1", {})
        _set_listing_state(s, "http://a", "discovered")
        _set_listing_state(s, "http://b", "ranked", score=0.8)
        assert _state_listings(s, "discovered") == ["http://a"]
        assert _state_listings(s, "ranked") == ["http://b"]

    def test_top_ranked_unshown(self):
        s = new_session("u1", {})
        _set_listing_state(s, "http://a", "ranked", score=0.5)
        _set_listing_state(s, "http://b", "ranked", score=0.9)
        assert _top_ranked_unshown(s) == "http://b"

    def test_unshown_listings(self):
        s = new_session("u1", {})
        _set_listing_state(s, "http://a", "ranked", score=0.5)
        _set_listing_state(s, "http://b", "shown")
        assert _unshown_listings(s) == ["http://a"]


class TestApplyFeedbackInline:
    def test_positive_weight_increase(self):
        prefs = {"ranking_weights": {}}
        signals = FeedbackSignals(verdict="maybe", positives=["natural_light"], raw_text="nice light")
        _apply_feedback_inline(prefs, signals, {"visual_summary": {}})
        assert prefs["ranking_weights"]["natural_light"] == pytest.approx(0.1)

    def test_negative_weight_decrease(self):
        prefs = {"ranking_weights": {}}
        signals = FeedbackSignals(verdict="no", negatives=["dark_rooms"], raw_text="too dark")
        _apply_feedback_inline(prefs, signals, {"visual_summary": {}})
        assert prefs["ranking_weights"]["dark_rooms"] == pytest.approx(-0.12)

    def test_decay_applied(self):
        prefs = {"ranking_weights": {"existing": 0.5}}
        signals = FeedbackSignals(verdict="neutral", raw_text="ok")
        _apply_feedback_inline(prefs, signals, {"visual_summary": {}})
        assert prefs["ranking_weights"]["existing"] == pytest.approx(0.49)
