# Iterative multi-page Zillow search orchestrator: implementation plan

This document is an engineering plan for extending the current single-page Zillow search into a continuous, multi-page, feedback-driven search loop with persistent memory, image understanding, and iterative ranking updates.

---

## Overview

The goal is an orchestrator that:

1. Searches Zillow across many pages, not just the first.
2. Shows the user one listing at a time and asks for feedback ("yes / no / maybe / comment").
3. Updates learned preference weights from every piece of feedback.
4. Re-ranks future listings based on those weights.
5. Triggers the existing contact flow (in `app/contact/contact_agent.py`) when the user says "yes" — but **keeps searching**.
6. Maintains durable user preferences and per-session state across restarts.

```
user transcript / criteria
   → search loop (paginated Zillow)
       → detail scraper       ← one listing at a time
           → image summarizer ← all photos from the page
       → listing dossier
   → ranking agent            ← dossier + user preference weights
   → present to user
       → feedback             → contact flow (on "yes")
                              → update preference weights
                              → continue to next listing / next page
```

---

## 1. State machine for the orchestrator

### Listing lifecycle states

Each listing discovered during a session moves through the following states:

```
discovered → analyzed → ranked → shown → [liked | rejected | deferred]
                                              ↓
                                    contact_requested → contacted
```

| State | Meaning |
|---|---|
| `discovered` | URL found on a Zillow search results page |
| `analyzed` | Detail page scraped; image summary and dossier built |
| `ranked` | Scored against current preference weights |
| `shown` | Presented to the user |
| `liked` | User said "yes" or "maybe" |
| `rejected` | User said "no" or gave negative feedback |
| `deferred` | User asked to revisit later |
| `contact_requested` | `run_contact_flow()` was invoked |
| `contacted` | Contact flow completed successfully |

### Transitions and triggers

```
discovered
  ↓  (detail scraper + image summarizer complete)
analyzed
  ↓  (ranking agent scores the dossier)
ranked
  ↓  (orchestrator picks the top-ranked unshown listing)
shown
  ↓  user says "no" / negative comment       → rejected
  ↓  user says "yes"                          → liked + contact_requested
  ↓  user says "maybe"                        → liked (no contact flow)
  ↓  user says "skip" or "come back later"    → deferred
```

**Side effects on "yes":**

1. Change listing state to `liked`, then `contact_requested`.
2. Call `run_contact_flow(listing_url, lead, mode="submit")` from `app/contact/contact_agent.py`.
3. Log `ContactResult` to session memory.
4. Update preference weights from positive signal.
5. **Continue searching** — do not stop the loop.

**Side effects on "no" / negative comment:**

1. Change listing state to `rejected`.
2. Extract structured negative signal via `extract_feedback`.
3. Update preference weights (soft penalties or hard-constraint flags).
4. Move on to the next ranked listing.

### Orchestrator loop (pseudocode)

```python
def run_search_loop(session: SearchSession):
    while not session.user_done:
        # 1. Refill the work queue if running low
        if len(session.unshown_listings()) < REFILL_THRESHOLD:
            page = session.next_page_cursor
            raw = paginated_search(session.criteria, page=page)
            session.add_discovered(raw.listing_links)
            session.next_page_cursor = page + 1

        # 2. Analyze unanalyzed listings (can be async/batched)
        for url in session.state_listings("discovered"):
            dossier = build_dossier(url, session.user_prefs)
            session.set_listing_state(url, "analyzed", dossier=dossier)

        # 3. Rank analyzed listings
        ranked = rank_listings(
            session.state_listings("analyzed"),
            session.user_prefs,
        )
        for url, score in ranked:
            session.set_listing_state(url, "ranked", score=score)

        # 4. Pick best unshown listing
        listing = session.top_ranked_unshown()
        if listing is None:
            continue   # wait for more to be analyzed

        session.set_listing_state(listing.url, "shown")
        feedback = present_to_user(listing)   # blocking: wait for user

        # 5. Process feedback
        signals = extract_feedback(feedback.text)
        update_preference_weights(session.user_prefs, signals, listing)

        if signals.verdict == "yes":
            session.set_listing_state(listing.url, "liked")
            result = run_contact_flow(listing.url, session.lead, mode="submit")
            session.record_contact(listing.url, result)
        elif signals.verdict in ("no", "negative"):
            session.set_listing_state(listing.url, "rejected")
        elif signals.verdict == "maybe":
            session.set_listing_state(listing.url, "liked")
```

---

## 2. Persistent memory schema

Memory is split into two tiers:

- **Durable user preferences** — survive across sessions; stored in a user profile.
- **Session / search state** — scoped to one search run; can be persisted for resume but are not permanent.

### 2.1 Durable user preferences

Stable things that the user has told us or that we have learned across many sessions.

```python
UserPreferences = {
    # Hard constraints (must match — violations disqualify listing)
    "hard_constraints": {
        "max_price": 3500,
        "min_beds": 2,
        "locations": ["Brooklyn, NY", "Astoria, NY"],
        "no_go_features": ["ground floor", "no laundry in unit"]
    },

    # Soft ranking weights (higher = more important to user)
    # Keys are feature/attribute names; values are floats in [-1.0, 1.0]
    "ranking_weights": {
        "natural_light": 0.9,
        "updated_kitchen": 0.7,
        "outdoor_space": 0.6,
        "high_floor": 0.5,
        "modern_bathroom": 0.4,
        "storage": 0.3,
        "dated_bathroom": -0.6,
        "dark_rooms": -0.8,
        "noisy_street": -0.5
    },

    # Explicit statements the user has made
    "stated_preferences": [
        "I love natural light",
        "updated kitchen is important",
        "I hate dark apartments"
    ],

    # Visual focus: things to look out for in images
    # Fed into the image summarizer as explicit attention prompts
    "visual_focus": [
        "check if the kitchen is updated",
        "look for natural light in every room",
        "flag if the bathroom looks dated",
        "check if there is outdoor space or a balcony"
    ],

    # Examples learned from feedback (used as few-shot context)
    "liked_examples": [
        {"url": "...", "reason": "bright, updated kitchen, balcony"}
    ],
    "rejected_examples": [
        {"url": "...", "reason": "dark rooms, dated bathroom"}
    ],

    # Metadata
    "created_at": "2025-01-01T00:00:00Z",
    "updated_at": "2025-01-15T12:00:00Z",
    "version": 3
}
```

### 2.2 Session / search state

Per-run state that tracks the current search in progress.

```python
SearchSession = {
    "session_id": "sess_abc123",
    "user_id": "user_xyz",
    "created_at": "2025-01-15T10:00:00Z",

    # Criteria used for this search
    "criteria": {
        "location": "Brooklyn NY",
        "intent": "rent",
        "price_max": 3500,
        "beds_min": 2,
        "misc_criteria": ["natural light", "updated kitchen"]
    },

    # Pagination cursor for Zillow result pages
    "next_page_cursor": 3,

    # Per-listing states
    "listings": {
        "https://zillow.com/...": {
            "state": "shown",
            "discovered_at": "2025-01-15T10:05:00Z",
            "score": 0.82,
            "dossier": { ... },   # see dossier schema below
            "feedback": {
                "raw_text": "I like the kitchen but the bathroom looks old",
                "signals": {
                    "verdict": "maybe",
                    "positives": ["updated_kitchen"],
                    "negatives": ["dated_bathroom"]
                }
            }
        }
    },

    # Contact actions taken this session
    "contact_log": [
        {
            "url": "https://zillow.com/...",
            "attempted_at": "2025-01-15T10:30:00Z",
            "result": { "submitted": true, "fields_filled": {...} }
        }
    ],

    # Feedback history in order (for context window injection)
    "feedback_history": [
        {
            "listing_url": "...",
            "feedback_text": "love the kitchen, bathroom is dated",
            "verdict": "maybe"
        }
    ]
}
```

### 2.3 Listing dossier schema

The compact intermediate object passed to the ranking agent.

```python
ListingDossier = {
    "url": "https://zillow.com/...",
    "listing_id": "2123456789",
    "address": "123 Main St, Brooklyn, NY",
    "price": 3200,
    "beds": 2,
    "baths": 1,
    "sqft": 950,
    "description_summary": "Sunny 2BR with updated kitchen...",
    "images": [
        {"url": "...", "position": 0},
        {"url": "...", "position": 1}
    ],
    "visual_summary": {
        "overall_condition": "good",
        "lighting": "bright",
        "style": ["updated kitchen", "hardwood floors"],
        "notable_features": [
            "stainless steel appliances",
            "large windows in living room"
        ],
        "possible_issues": [
            "bathroom appears dated",
            "bedroom seems compact"
        ],
        "uncertainties": [
            "no clear exterior shot"
        ],
        "representative_images": [
            {"url": "...", "room_type": "kitchen"},
            {"url": "...", "room_type": "living_room"},
            {"url": "...", "room_type": "bedroom"},
            {"url": "...", "room_type": "bathroom"}
        ]
    },
    "facts": {
        "pet_friendly": true,
        "laundry": "in unit",
        "parking": "none"
    }
}
```

---

## 3. Feedback-driven ranking weight updates

### 3.1 Signal types

| Type | Examples | How extracted |
|---|---|---|
| Explicit verdict | "yes", "no", "maybe", "skip" | `extract_feedback` parses keywords |
| Explicit attribute | "I love the kitchen", "bathroom is dated" | NLP / structured extraction |
| Implicit positive | User asks to contact → strong positive | Derived from action |
| Implicit negative | User skips without comment | Weak negative signal |

### 3.2 Hard constraints vs soft weights

**Hard constraints** are binary disqualifiers. If a listing violates a hard constraint it is removed before ranking, not just scored lower.

Examples: `price > max_price`, `beds < min_beds`, feature in `no_go_features`.

Hard constraints are updated only on explicit, confident user statements such as:
- "I will never rent on a ground floor" → add `"ground floor"` to `no_go_features`.
- "I need at least 2 bedrooms" → set `min_beds = 2`.

**Soft weights** are floating-point scores in `[-1.0, 1.0]` that shift the ranking. They are updated from both explicit and inferred feedback.

### 3.3 Weight update logic

```python
def update_preference_weights(
    prefs: UserPreferences,
    signals: FeedbackSignals,
    listing: ListingDossier,
    *,
    positive_step: float = 0.1,
    negative_step: float = 0.12,
    saturation_cap: float = 1.0,
    decay_factor: float = 0.02,
) -> None:
    """
    Adjust soft ranking_weights based on feedback signals from one interaction.

    - positive_step: how much to increase a weight on positive evidence
    - negative_step: how much to decrease a weight on negative evidence
      (slightly higher because negative feedback tends to be more decisive)
    - saturation_cap: weights are clamped to [-saturation_cap, +saturation_cap]
    - decay_factor: all weights decay slightly toward 0 each update to prevent
      stale strong opinions from dominating forever
    """
    weights = prefs["ranking_weights"]

    # 1. Decay all weights slightly toward zero
    for key in list(weights.keys()):
        weights[key] *= (1.0 - decay_factor)

    # 2. Apply positive signals
    for feature in signals.positives:
        current = weights.get(feature, 0.0)
        weights[feature] = min(current + positive_step, saturation_cap)

    # 3. Apply negative signals
    for feature in signals.negatives:
        current = weights.get(feature, 0.0)
        weights[feature] = max(current - negative_step, -saturation_cap)

    # 4. Strong positive (verdict == "yes") amplifies all positive features
    #    that appear in the listing's visual summary
    if signals.verdict == "yes":
        for feature in _features_from_dossier(listing):
            current = weights.get(feature, 0.0)
            weights[feature] = min(current + positive_step * 0.5, saturation_cap)

    prefs["updated_at"] = utcnow_iso()
```

### 3.4 Example weight updates

**Before feedback:**
```json
{
  "natural_light": 0.5,
  "updated_kitchen": 0.3,
  "dated_bathroom": -0.3
}
```

**User says:** `"I like the kitchen but the bathroom looks really old"`

**Extracted signals:**
```python
signals.positives = ["updated_kitchen"]
signals.negatives = ["dated_bathroom"]
signals.verdict = "maybe"
```

**After update:**
```json
{
  "natural_light": 0.49,        ← decayed slightly
  "updated_kitchen": 0.39,      ← +0.1, then capped
  "dated_bathroom": -0.41       ← -0.12
}
```

**User says:** `"yes"` to a listing with bright rooms and a renovated kitchen

**After update:**
```json
{
  "natural_light": 0.64,        ← indirect amplification from "yes"
  "updated_kitchen": 0.54,
  "dated_bathroom": -0.40       ← decayed
}
```

### 3.5 Confidence and feedback history

Weights that have been reinforced many times are more confident. Optionally track counts alongside weights:

```python
"ranking_weights": {
    "natural_light": {"value": 0.9, "confidence": 0.8, "updates": 7},
    "updated_kitchen": {"value": 0.7, "confidence": 0.6, "updates": 4}
}
```

Higher-confidence weights decay more slowly and require stronger contradicting evidence to shift. This prevents a single outlier interaction from reversing a well-established preference.

### 3.6 Updating visual focus from feedback

When a new feature becomes highly weighted, it should be added to `visual_focus` so the image summarizer pays attention to it in future listings:

```python
def sync_visual_focus(prefs: UserPreferences) -> None:
    """Ensure high-weight features appear in visual_focus."""
    for feature, weight in prefs["ranking_weights"].items():
        if weight > 0.6 and feature not in prefs["visual_focus"]:
            prefs["visual_focus"].append(f"check if listing has: {feature}")
        elif weight < -0.5:
            prefs["visual_focus"].append(f"flag if listing shows: {feature}")
```

---

## 4. Multi-page Zillow search plan

### 4.1 Current behavior

`data/zillow/scraper.py` builds a single Zillow URL, fetches it with Playwright, parses results, and returns listings + links. It has no pagination.

### 4.2 Pagination approach

Zillow search pages use a query parameter for pagination. The standard pattern is:

```
https://www.zillow.com/homes/for_rent/Brooklyn-NY/?searchQueryState={"pagination":{"currentPage":2},...}
```

For an MVP, appending `&page=N` or adjusting `currentPage` in the `searchQueryState` JSON is enough to step through pages.

New module: **`data/zillow/paginated_search.py`**

```python
def paginated_search(criteria: dict, page: int = 1) -> dict:
    """
    Fetch one page of Zillow results.
    Returns the same shape as scraper.search() plus a `page` field.
    """
    url = build_search_url(criteria, page=page)
    raw_html = fetch_html(url)
    listings, listing_links = parse_listings(raw_html)
    return {
        "page": page,
        "listings": listings,
        "listing_links": listing_links,
        "raw_html": raw_html,
        "search_url": url,
    }
```

Update `build_search_url()` in `scraper.py` to accept an optional `page` argument and inject the pagination parameter.

### 4.3 New module map

| Module | Responsibility |
|---|---|
| `data/zillow/paginated_search.py` | Fetch one page; accept `page=N` |
| `data/zillow/detail_scraper.py` | Scrape a single listing detail page; return all image URLs (ordered), description, facts |
| `app/agents/extract_preferences.py` | Parse user transcript / chat into initial `UserPreferences` |
| `app/agents/derive_visual_focus.py` | Turn `ranking_weights` and stated preferences into `visual_focus` prompts |
| `app/agents/summarize_listing_images.py` | Input: listing detail + visual_focus; output: `visual_summary` |
| `app/agents/rank_listings.py` | Score each dossier against `UserPreferences`; return sorted list |
| `app/agents/extract_feedback.py` | Parse user feedback text into structured `FeedbackSignals` |
| `app/memory/user_preference_memory.py` | Load/save `UserPreferences`; apply weight updates |
| `app/orchestrators/search_loop.py` | Main loop; manages `SearchSession`; coordinates all agents |

---

## 5. Detail scraper

New module: **`data/zillow/detail_scraper.py`**

Its job is to open a single listing page and return all the raw data faithfully. The image summarizer handles what to do with the images.

```python
def scrape_listing_detail(url: str) -> dict:
    """
    Fetch and parse a single Zillow listing page.
    Returns all image URLs (in page order), description, price,
    beds/baths/sqft, address, and structured facts.
    """
    raw_html = fetch_html(url)
    return {
        "url": url,
        "address": ...,
        "price": ...,
        "beds": ...,
        "baths": ...,
        "sqft": ...,
        "description": ...,
        "images": [
            {"url": img_url, "position": i}
            for i, img_url in enumerate(extract_all_images(raw_html))
        ],
        "facts": extract_facts(raw_html),
    }
```

All image URLs present on the page are collected and ordered by position. Exact-duplicate URLs are removed. The scraper does **not** filter or classify images — that is the summarizer's job.

---

## 6. Image summarizer and ranking integration

### 6.1 Image summarizer agent

Module: **`app/agents/summarize_listing_images.py`**

**Inputs:**
- Listing detail dict from `detail_scraper`
- `visual_focus` list from user preferences (things to specifically look for)

**Process:**

```
all images (N photos)
  → filter obvious junk (maps, logos, floorplans) by URL/heuristic
  → exact-dedupe URLs
  → pass first 12–15 images to vision model (high priority)
  → produce: room classifications + observations per image
  → aggregate to listing-level visual summary
  → if listing becomes a finalist: pass remaining images for extra confidence
```

**The visual_focus parameter** turns user preferences into explicit attention prompts. For example if `visual_focus` contains `"check if the kitchen is updated"`, the summarizer's prompt includes that instruction so it pays extra attention to kitchen images and flags evidence either way.

Example prompt fragment:
```
You are reviewing photos of a rental listing.
Please summarize what you observe across all photos.
Pay special attention to:
- check if the kitchen is updated
- look for natural light in every room
- flag if the bathroom looks dated
- check if there is outdoor space or a balcony

Return a JSON object with the schema: { visual_summary: ... }
```

**Output:** `visual_summary` dict as shown in the dossier schema above.

### 6.2 Dossier builder

Module: **`app/orchestrators/search_loop.py`** (or a helper `build_dossier()` function)

```python
def build_dossier(url: str, user_prefs: UserPreferences) -> ListingDossier:
    detail = scrape_listing_detail(url)
    visual_summary = summarize_listing_images(
        detail,
        visual_focus=user_prefs["visual_focus"]
    )
    description_summary = summarize_description(detail["description"])
    return {**detail, "visual_summary": visual_summary, "description_summary": description_summary}
```

### 6.3 Ranking agent

Module: **`app/agents/rank_listings.py`**

The ranking agent receives a list of dossiers and the current user preferences. It scores each listing against the preference weights and hard constraints.

```python
def rank_listings(dossiers: list[dict], user_prefs: UserPreferences) -> list[tuple[str, float]]:
    """
    Returns list of (url, score) sorted descending.
    Listings that violate hard constraints are excluded.
    """
    scored = []
    for dossier in dossiers:
        if violates_hard_constraints(dossier, user_prefs["hard_constraints"]):
            continue
        score = score_dossier(dossier, user_prefs["ranking_weights"])
        scored.append((dossier["url"], score))
    return sorted(scored, key=lambda x: x[1], reverse=True)
```

The score aggregates:
- Price fit (how far under `max_price`)
- Match between `visual_summary` features and positive `ranking_weights`
- Penalty for features matching negative `ranking_weights`
- Bonus for features matching `liked_examples` patterns
- Penalty for features matching `rejected_examples` patterns

---

## 7. Feedback extraction

Module: **`app/agents/extract_feedback.py`**

Turns raw user text into structured signals.

```python
@dataclass
class FeedbackSignals:
    verdict: str          # "yes" | "no" | "maybe" | "skip" | "neutral"
    positives: list[str]  # feature names that the user liked
    negatives: list[str]  # feature names that the user disliked
    hard_constraint_updates: list[dict]  # any new hard constraints stated
    raw_text: str
```

The LLM prompt maps phrases to known feature keys:
- `"love the kitchen"` → `positives: ["updated_kitchen"]`
- `"too dark"` → `negatives: ["dark_rooms"]`
- `"bathroom outdated"` → `negatives: ["dated_bathroom"]`
- `"perfect"` / `"yes"` → `verdict: "yes"`
- `"not for me"` / `"no"` → `verdict: "no"`
- `"I'll never live on the ground floor"` → `hard_constraint_updates: [{"type": "add_no_go", "value": "ground floor"}]`

---

## 8. Preference memory module

Module: **`app/memory/user_preference_memory.py`**

Handles load/save of `UserPreferences` and exposes the weight update function.

```python
class UserPreferenceMemory:
    def __init__(self, storage_path: str): ...
    def load(self, user_id: str) -> UserPreferences: ...
    def save(self, user_id: str, prefs: UserPreferences) -> None: ...
    def apply_feedback(self, prefs: UserPreferences, signals: FeedbackSignals, listing: ListingDossier) -> None: ...
    def sync_visual_focus(self, prefs: UserPreferences) -> None: ...
```

Storage can be a JSON file per user (MVP) or a database row (production).

---

## 9. Phased implementation plan

### Phase 1 — MVP (single session, no persistence)

Goal: search multiple pages, show listings, trigger contact on "yes", keep going.

- [ ] Add `page` param to `build_search_url()` in `scraper.py`
- [ ] Create `data/zillow/paginated_search.py`
- [ ] Create `data/zillow/detail_scraper.py` (scrape images, description, facts)
- [ ] Create `app/agents/summarize_listing_images.py` (first 10–15 images, `visual_focus` param)
- [ ] Create `app/agents/rank_listings.py` (simple scoring over dossier + weights)
- [ ] Create `app/agents/extract_feedback.py` (verdict + positives/negatives)
- [ ] Create `app/orchestrators/search_loop.py` (in-memory session, no disk persistence)
- [ ] Wire contact flow: on "yes" call `run_contact_flow()` then continue loop

### Phase 2 — Persistent memory

- [ ] Create `app/memory/user_preference_memory.py`
- [ ] Persist `UserPreferences` to JSON file (keyed by user/phone)
- [ ] Persist `SearchSession` so search can be resumed after restart
- [ ] Load saved `liked_examples` / `rejected_examples` as few-shot context in ranking agent

### Phase 3 — Richer feedback and weights

- [ ] Create `app/agents/extract_preferences.py` (initial preference extraction from onboarding transcript)
- [ ] Create `app/agents/derive_visual_focus.py` (auto-derive visual_focus from ranking_weights)
- [ ] Add confidence tracking to ranking weights
- [ ] Add decay/saturation/clamping to weight updates
- [ ] Sync visual_focus automatically after each weight update

### Phase 4 — Image pipeline improvements

- [ ] Add image deduplication (perceptual hash or URL-based)
- [ ] Add room-type classifier (vision model or prompt)
- [ ] Add progressive disclosure: full pass only on finalist listings
- [ ] Cache image summaries by listing URL + image hash to avoid reprocessing

### Phase 5 — Production hardening

- [ ] Replace JSON file storage with a database
- [ ] Add rate limiting / retry logic for Zillow pagination
- [ ] Add observability: log per-listing scores, feedback signals, weight history
- [ ] Expose search progress to frontend via streaming or polling endpoint

---

## 10. Flow in one go

```
user message / criteria
  → extract_preferences()           → UserPreferences (initial weights + visual_focus)
  → search_loop starts

  loop:
    paginated_search(criteria, page=N)
      → listing URLs discovered
    for each new URL:
      detail_scraper(url)
        → images[], description, facts
      summarize_listing_images(detail, visual_focus)
        → visual_summary
      build_dossier()
        → ListingDossier

    rank_listings(dossiers, user_prefs)
      → sorted [(url, score), ...]

    present top listing to user
    ← user feedback text

    extract_feedback(text)
      → FeedbackSignals { verdict, positives, negatives }

    if verdict == "yes":
      run_contact_flow(url, lead, mode="submit")   ← from app/contact/contact_agent.py
      record ContactResult in session

    update_preference_weights(user_prefs, signals, listing)
    sync_visual_focus(user_prefs)
    save user_prefs

    continue loop (next best unshown listing)
    if queue low: fetch next Zillow page
```

Files: `data/zillow/paginated_search.py` (pagination), `data/zillow/detail_scraper.py` (listing detail + images), `app/agents/summarize_listing_images.py` (visual compression), `app/agents/rank_listings.py` (scoring), `app/agents/extract_feedback.py` (signal extraction), `app/agents/extract_preferences.py` (onboarding), `app/agents/derive_visual_focus.py` (attention prompts), `app/memory/user_preference_memory.py` (load/save/update weights), `app/orchestrators/search_loop.py` (main loop).
