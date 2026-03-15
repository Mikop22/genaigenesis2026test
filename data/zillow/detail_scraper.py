"""Scrape a single Zillow listing detail page for images, description, and facts."""

import json
import re

from bs4 import BeautifulSoup

from .playwright_fetch import fetch_html

DETAIL_SELECTOR = "[data-testid='bed-bath-beyond'], .ds-price, .hdp-content-wrapper"

# URL patterns that are typically non-photo assets (maps, logos, floorplans)
_JUNK_IMAGE_PATTERNS = re.compile(
    r"(maps\.googleapis|logo|brand|icon|floorplan|streetview|\.svg)", re.I
)


def _is_photo_url(url: str) -> bool:
    """Return True if the URL looks like an actual listing photo."""
    if not url or not url.startswith("http"):
        return False
    return not _JUNK_IMAGE_PATTERNS.search(url)


def _extract_all_images(html: str) -> list[str]:
    """Return ordered, deduplicated image URLs from the listing page."""
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()

    # Try structured data first (JSON-LD or embedded script)
    for script in soup.select("script[type='application/ld+json']"):
        try:
            data = json.loads(script.string or "")
            for img in data.get("image", []) if isinstance(data, dict) else []:
                u = img if isinstance(img, str) else img.get("url", "")
                if u and u not in seen and _is_photo_url(u):
                    seen.add(u)
                    urls.append(u)
        except (json.JSONDecodeError, TypeError):
            pass

    # Carousel / gallery images
    for img in soup.select("img[src]"):
        src = img.get("src", "").strip()
        if src and src not in seen and _is_photo_url(src):
            seen.add(src)
            urls.append(src)

    # Also try srcset for higher-res versions
    for img in soup.select("img[srcset]"):
        for part in (img.get("srcset") or "").split(","):
            src = part.strip().split()[0] if part.strip() else ""
            if src and src not in seen and _is_photo_url(src):
                seen.add(src)
                urls.append(src)

    return urls


def _extract_facts(soup: BeautifulSoup) -> dict:
    """Extract structured facts (pet policy, laundry, parking, etc.)."""
    facts: dict[str, str] = {}
    # Zillow fact items are often in <li> under .hdp-facts-list or similar
    for li in soup.select(".fact-value, [data-testid] li"):
        text = li.get_text(strip=True)
        lower = text.lower()
        if "pet" in lower:
            facts["pet_friendly"] = text
        elif "laundry" in lower:
            facts["laundry"] = text
        elif "parking" in lower:
            facts["parking"] = text
        elif "year" in lower and "built" in lower:
            facts["year_built"] = text
        elif "cooling" in lower or "heating" in lower:
            facts["climate_control"] = text
    return facts


def _extract_numeric(text: str) -> str:
    """Pull numeric portion from a string like '$3,200/mo' → '3200'."""
    nums = re.sub(r"[^0-9.]", "", text)
    return nums.split(".")[0] if nums else ""


def scrape_listing_detail(
    url: str,
    *,
    headless: bool = False,
) -> dict:
    """
    Fetch and parse a single Zillow listing page.

    Returns all image URLs (in page order), description, price,
    beds/baths/sqft, address, and structured facts.
    """
    html = fetch_html(
        url,
        headless=headless,
        wait_selector=DETAIL_SELECTOR,
        wait_timeout_ms=25_000,
        pause_for_captcha=True,
    )
    if not html:
        return {
            "url": url,
            "address": "",
            "price": "",
            "beds": "",
            "baths": "",
            "sqft": "",
            "description": "",
            "images": [],
            "facts": {},
        }

    soup = BeautifulSoup(html, "html.parser")

    # Address
    addr_el = soup.select_one("h1, [data-testid='bdp-address']")
    address = addr_el.get_text(strip=True) if addr_el else ""

    # Price
    price_el = soup.select_one(".ds-price, [data-testid='price'], .summary-container .price")
    price_text = price_el.get_text(strip=True) if price_el else ""

    # Beds / baths / sqft from bed-bath-beyond or similar
    def _stat(pattern: str) -> str:
        el = soup.find(string=re.compile(pattern, re.I))
        if el:
            parent = el.find_parent()
            if parent:
                return parent.get_text(strip=True)
        return ""

    beds = _stat(r"\bbed")
    baths = _stat(r"\bbath")
    sqft = _stat(r"\bsqft|\bsq\.?\s*ft")

    # Description
    desc_el = soup.select_one("[data-testid='description'], .ds-overview .Text")
    description = desc_el.get_text(strip=True) if desc_el else ""

    images = _extract_all_images(html)
    facts = _extract_facts(soup)

    return {
        "url": url,
        "address": address,
        "price": _extract_numeric(price_text) if price_text else "",
        "beds": _extract_numeric(beds) if beds else "",
        "baths": _extract_numeric(baths) if baths else "",
        "sqft": _extract_numeric(sqft) if sqft else "",
        "description": description,
        "images": [
            {"url": img_url, "position": i}
            for i, img_url in enumerate(images)
        ],
        "facts": facts,
    }
