"""Zillow search: build URL, fetch via Playwright, parse."""
from urllib.parse import quote_plus

from .parse import dedupe_links, listing_links_from_html, parse_listings
from .playwright_fetch import fetch_html

SEARCH_SELECTOR = "script[data-zrr-shared-data-key], article, [data-test='property-card-price']"


def build_search_url(criteria: dict) -> str:
    location = criteria.get("location", "")
    intent = criteria.get("intent", "rent")
    # Normalize "buy" to "sale" so the URL is always valid on Zillow
    if intent == "buy":
        intent = "sale"
    price_max = criteria.get("price_max", "")
    beds_min = criteria.get("beds_min", "")
    slug = quote_plus(location)
    return f"https://www.zillow.com/homes/for_{intent}/{slug}/?price_max={price_max}&beds_min={beds_min}"


def _coerce_misc_criteria(value) -> list[str]:
    """Ensure misc_criteria is always a list of strings regardless of LLM output shape."""
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str) and value:
        return [value]
    return []


def search(criteria: dict, *, headless: bool = False) -> dict:
    url = build_search_url(criteria)
    html = fetch_html(
        url,
        headless=headless,
        wait_selector=SEARCH_SELECTOR,
        wait_timeout_ms=25_000,
        pause_for_captcha=True,
    )
    misc_criteria = _coerce_misc_criteria(criteria.get("misc_criteria"))
    if not html:
        return {"listings": [], "listing_links": [], "raw_html": "", "search_url": url, "misc_criteria": misc_criteria}
    listings = parse_listings(html)
    links = listing_links_from_html(html)
    if not links and listings:
        links = [r.get("url", "") or "" for r in listings]
    links = dedupe_links(links)
    return {
        "listings": listings,
        "listing_links": links,
        "raw_html": html,
        "search_url": url,
        "misc_criteria": misc_criteria,
    }
