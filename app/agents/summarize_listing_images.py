"""Summarize listing images using a vision-capable LLM with user-specific visual focus."""

import json
import logging
import os
from typing import Optional

LOG = logging.getLogger(__name__)


def _build_image_summary_prompt(visual_focus: list[str]) -> str:
    """Build the system prompt for the image summarizer."""
    focus_lines = ""
    if visual_focus:
        items = "\n".join(f"- {f}" for f in visual_focus)
        focus_lines = f"\nPay special attention to:\n{items}\n"

    return (
        "You are reviewing photos of a rental/sale listing.\n"
        "Please summarize what you observe across all photos.\n"
        f"{focus_lines}\n"
        "Return a JSON object with this exact schema:\n"
        "{\n"
        '  "overall_condition": "<good|fair|poor>",\n'
        '  "lighting": "<bright|moderate|dark>",\n'
        '  "style": ["<feature1>", "<feature2>"],\n'
        '  "notable_features": ["<feature1>", "<feature2>"],\n'
        '  "possible_issues": ["<issue1>", "<issue2>"],\n'
        '  "uncertainties": ["<note1>"],\n'
        '  "representative_images": [\n'
        '    {"url": "<url>", "room_type": "<kitchen|living_room|bedroom|bathroom|other>"}\n'
        "  ]\n"
        "}\n"
        "Only return the JSON object, no other text."
    )


def _default_visual_summary() -> dict:
    """Fallback visual summary when the LLM call fails or is unavailable."""
    return {
        "overall_condition": "unknown",
        "lighting": "unknown",
        "style": [],
        "notable_features": [],
        "possible_issues": [],
        "uncertainties": ["image analysis unavailable"],
        "representative_images": [],
    }


def _parse_visual_summary(text: str) -> dict:
    """Parse the LLM response into a visual summary dict."""
    text = (text or "").strip()
    if "```" in text:
        start = text.find("```")
        if "json" in text[: start + 10]:
            start = text.find("\n", start) + 1
        end = text.find("```", start)
        text = text[start:end] if end > start else text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return _default_visual_summary()


def summarize_listing_images(
    detail: dict,
    visual_focus: Optional[list[str]] = None,
    *,
    max_images: int = 15,
) -> dict:
    """
    Summarize listing images using a vision model.

    Parameters
    ----------
    detail : dict
        Listing detail from ``detail_scraper.scrape_listing_detail()``.
    visual_focus : list[str], optional
        User-specific things to look for (e.g. "check if kitchen is updated").
    max_images : int
        Maximum number of images to send to the vision model.

    Returns
    -------
    dict
        Visual summary with keys: overall_condition, lighting, style,
        notable_features, possible_issues, uncertainties, representative_images.
    """
    images = detail.get("images", [])[:max_images]
    if not images:
        return _default_visual_summary()

    system_prompt = _build_image_summary_prompt(visual_focus or [])

    image_descriptions = []
    for img in images:
        image_descriptions.append(f"Image {img['position']}: {img['url']}")
    user_content = (
        f"Listing: {detail.get('address', 'Unknown address')}\n"
        f"Description: {detail.get('description', 'No description')}\n\n"
        f"Photos:\n" + "\n".join(image_descriptions)
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
                    {"role": "user", "content": user_content},
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _parse_visual_summary(text)
    except Exception as exc:
        LOG.warning("Image summarization failed: %s", exc)
        return _default_visual_summary()
