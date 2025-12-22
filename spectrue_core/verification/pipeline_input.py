# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
from __future__ import annotations

import logging
import re

from spectrue_core.utils.trace import Trace
from spectrue_core.utils.url_utils import get_registrable_domain

logger = logging.getLogger(__name__)


def is_url_input(text: str) -> bool:
    """
    True if the user input should be treated as a URL (must start with http(s)).
    """
    if not text or len(text) > 500:
        return False
    stripped = text.strip()
    return stripped.startswith("http://") or stripped.startswith("https://")


async def resolve_url_content(search_mgr, url: str, *, log: logging.Logger = logger) -> str | None:
    """Fetch URL content via Tavily Extract. Cleaning happens in claim extraction."""
    try:
        raw_text = await search_mgr.fetch_url_content(url)
        if not raw_text or len(raw_text) < 50:
            return None

        log.info("[Pipeline] URL resolved: %d chars", len(raw_text))
        Trace.event("pipeline.url_resolved", {"original": url, "chars": len(raw_text)})
        return raw_text

    except Exception as e:
        log.warning("[Pipeline] Failed to resolve URL: %s", e)
        return None


def extract_url_anchors(text: str, *, exclude_url: str | None = None) -> list[dict]:
    """Extract URL-anchor pairs from article text."""
    if not text:
        return []

    anchors: list[dict] = []
    exclude_domain = get_registrable_domain(exclude_url) if exclude_url else None
    seen_domains: set[str] = set()

    md_pattern = r"\[([^\]]+)\]\((https?://[^\s\)]+)\)"
    for match in re.finditer(md_pattern, text):
        anchor, url = match.groups()
        url = url.rstrip(".,;:!?")
        domain = get_registrable_domain(url)

        if exclude_domain and domain == exclude_domain:
            continue
        if domain in seen_domains:
            continue

        seen_domains.add(domain)
        anchors.append({"url": url, "anchor": anchor.strip(), "domain": domain})

    paren_pattern = r"([^(\n]{3,50})\s*\((https?://[^\s\)]+)\)"
    for match in re.finditer(paren_pattern, text):
        anchor, url = match.groups()
        url = url.rstrip(".,;:!?")
        domain = get_registrable_domain(url)

        if exclude_domain and domain == exclude_domain:
            continue
        if domain in seen_domains:
            continue

        anchor = anchor.strip()
        anchor = re.sub(r"^[^\w]*", "", anchor)
        if len(anchor) < 3:
            continue

        seen_domains.add(domain)
        anchors.append({"url": url, "anchor": anchor[:50], "domain": domain})

    url_pattern = r"https?://[^\s\]\)\}>'\"<,]+"
    for match in re.finditer(url_pattern, text):
        url = match.group().rstrip(".,;:!?")
        domain = get_registrable_domain(url)

        if exclude_domain and domain == exclude_domain:
            continue
        if domain in seen_domains:
            continue

        start = max(0, match.start() - 60)
        prefix = text[start : match.start()]
        anchor = prefix.split("\n")[-1].strip()
        anchor = re.sub(r"^[^\w]*", "", anchor)
        if len(anchor) < 5:
            anchor = domain

        seen_domains.add(domain)
        anchors.append({"url": url, "anchor": anchor[:50], "domain": domain})

    return anchors[:10]


def restore_urls_from_anchors(cleaned_text: str, url_anchors: list[dict]) -> list[dict]:
    """Find which URL anchors survived cleaning and return them as sources."""
    if not cleaned_text or not url_anchors:
        return []

    sources: list[dict] = []
    cleaned_lower = cleaned_text.lower()

    for item in url_anchors:
        anchor = item.get("anchor", "")
        url = item.get("url", "")
        domain = item.get("domain", "")

        if not anchor or not url:
            continue

        anchor_lower = anchor.lower()
        if anchor_lower in cleaned_lower:
            display_title = anchor if len(anchor) >= 10 else f"Джерело: {domain}"
            sources.append(
                {
                    "url": url,
                    "title": display_title,
                    "domain": domain,
                    "source_type": "inline",
                    "is_trusted": False,
                }
            )
            continue

        anchor_words = [w for w in anchor_lower.split() if len(w) > 3]
        if anchor_words:
            matches = sum(1 for w in anchor_words if w in cleaned_lower)
            if matches >= len(anchor_words) * 0.7:
                display_title = anchor if len(anchor) >= 10 else f"Джерело: {domain}"
                sources.append(
                    {
                        "url": url,
                        "title": display_title,
                        "domain": domain,
                        "source_type": "inline",
                        "is_trusted": False,
                    }
                )

    return sources[:5]

