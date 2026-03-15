"""Paginated Zillow search: fetch one page of results at a time."""

from .parse import dedupe_links, listing_links_from_html, parse_listings
from .playwright_fetch import fetch_html
from .scraper import SEARCH_SELECTOR, build_search_url


def paginated_search(
    criteria: dict,
    page: int = 1,
    *,
    headless: bool = False,
) -> dict:
    """
    Fetch one page of Zillow results.

    Returns the same shape as ``scraper.search()`` plus a ``page`` field.
    """
    url = build_search_url(criteria, page=page)
    html = fetch_html(
        url,
        headless=headless,
        wait_selector=SEARCH_SELECTOR,
        wait_timeout_ms=25_000,
        pause_for_captcha=True,
    )
    if not html:
        return {
            "page": page,
            "listings": [],
            "listing_links": [],
            "raw_html": "",
            "search_url": url,
        }
    listings = parse_listings(html)
    links = listing_links_from_html(html)
    if not links and listings:
        links = [r.get("url", "") or "" for r in listings]
    links = dedupe_links(links)
    return {
        "page": page,
        "listings": listings,
        "listing_links": links,
        "raw_html": html,
        "search_url": url,
    }
