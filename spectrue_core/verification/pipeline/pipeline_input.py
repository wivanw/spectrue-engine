# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

from spectrue_core.runtime_config import ContentBudgetConfig
from spectrue_core.analysis.content_budgeter import ContentBudgeter, TrimResult

from spectrue_core.utils.trace import Trace
from spectrue_core.utils.url_utils import get_registrable_domain

logger = logging.getLogger(__name__)


async def verify_inline_sources(
    *,
    inline_sources: List[dict],
    claims: List[Any],
    fact: str,
    agent: Any,
    search_mgr: Any = None,
    content_budget_config: Optional[ContentBudgetConfig] = None,
    progress_callback: Optional[Callable] = None,
) -> List[dict]:
    """
    Verify inline sources against claims.
    
    Args:
        inline_sources: List of source dicts
        claims: List of claim objects
        fact: The verification fact/text
        agent: FactCheckerAgent for relevance check
        search_mgr: SearchManager for ladder and fetching
        content_budget_config: Config for budgeting fetched content
        progress_callback: Async callback for progress
        
    Returns:
        List of verified inline sources (added to final_sources)
    """
    if not inline_sources or not claims:
        return []

    if progress_callback:
        await progress_callback("verifying_sources")

    # Check if inline source verification is disabled via feature flag
    runtime = getattr(agent, "runtime", None)
    enable_verification = True
    if runtime and hasattr(runtime, "llm"):
        enable_verification = getattr(runtime.llm, "enable_inline_source_verification", True)

    # If verification disabled, mark all sources as verification_skipped
    if not enable_verification:
        verified_inline_sources = []
        for src in inline_sources:
            src["is_primary"] = False
            src["is_relevant"] = True
            src["verification_skipped"] = True
            verified_inline_sources.append(src)

        Trace.event(
            "pipeline.inline_sources_verified",
            {
                "total": len(inline_sources),
                "passed": len(verified_inline_sources),
                "primary": 0,
                "verification_skipped": True,
            },
        )
        logger.debug("[Pipeline] Inline source verification disabled, marking %d sources as skipped", len(verified_inline_sources))
        return verified_inline_sources

    article_excerpt = fact[:500] if fact else ""
    verified_inline_sources = []

    verification_tasks = [
        agent.verify_inline_source_relevance(claims, src, article_excerpt)
        for src in inline_sources
    ]
    verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)

    for src, result in zip(inline_sources, verification_results):
        if isinstance(result, Exception):
            logger.warning("[Pipeline] Inline source verification failed: %s", result)
            src["is_primary"] = False
            src["is_relevant"] = True
            src["verification_skipped"] = True  # Mark as skipped due to error
            verified_inline_sources.append(src)
            continue

        is_relevant = result.get("is_relevant", True)
        is_primary = result.get("is_primary", False)
        verification_skipped = result.get("verification_skipped", False)

        if not is_relevant:
            logger.debug("[Pipeline] Inline source rejected: %s", src.get("domain"))
            continue

        src["is_primary"] = is_primary
        src["is_relevant"] = True
        if verification_skipped:
            src["verification_skipped"] = True
        if is_primary:
            src["is_trusted"] = True
        verified_inline_sources.append(src)
        logger.debug(
            "[Pipeline] Inline source %s: relevant=%s, primary=%s",
            src.get("domain"),
            is_relevant,
            is_primary,
        )

    if verified_inline_sources:
        if search_mgr and hasattr(search_mgr, "apply_evidence_acquisition_ladder"):
            verified_inline_sources = await search_mgr.apply_evidence_acquisition_ladder(
                verified_inline_sources,
                budget_context="inline",  # Use separate budget from claim verification
            )
        
        skipped_count = len([s for s in verified_inline_sources if s.get("verification_skipped")])
        Trace.event(
            "pipeline.inline_sources_verified",
            {
                "total": len(inline_sources),
                "passed": len(verified_inline_sources),
                "primary": len([s for s in verified_inline_sources if s.get("is_primary")]),
                "verification_skipped": skipped_count > 0,
                "skipped_count": skipped_count,
            },
        )

        # Fetch content for primary inline sources (only if verification was done)
        primary_sources = [s for s in verified_inline_sources if s.get("is_primary") and not s.get("verification_skipped")]
        if primary_sources and search_mgr and content_budget_config:
            for src in primary_sources[:2]:  # Limit to 2 primary sources
                url = src.get("url", "")
                if url and not src.get("content"):
                    try:
                        fetched = await search_mgr.web_tool.fetch_page_content(url)
                        if fetched and len(fetched) > 100:
                            budgeted_content, _ = apply_content_budget(
                                fetched, content_budget_config, source="inline_source"
                            )
                            src["content"] = budgeted_content
                            src["snippet"] = budgeted_content[:300]
                            Trace.event("inline_source.content_fetched", {
                                "url": url[:80],
                                "chars": len(fetched),
                            })
                    except Exception as e:
                        logger.debug("[Pipeline] Failed to fetch inline source: %s", e)

    return verified_inline_sources



def apply_content_budget(
    text: str,
    config: ContentBudgetConfig,
    source: str = "unknown"
) -> Tuple[str, Optional[TrimResult]]:
    """
    Apply deterministic content budgeting to plain text.
    
    Args:
        text: Raw text to trim
        config: Budget configuration
        source: Source identifier for tracing events
        
    Returns:
        Tuple of (trimmed_text, TrimResult or None)
    """
    if not text:
        return text, None
        
    raw_len = len(text)
    if raw_len > int(config.absolute_guardrail_chars):
        Trace.event(
            "content_budgeter.guardrail",
            {
                "raw_len": raw_len,
                "absolute_guardrail_chars": int(config.absolute_guardrail_chars),
                "source": source
            },
        )
        raise ValueError("Input too large to process safely")

    if raw_len <= int(config.max_clean_text_chars_default):
        return text, None

    budgeter = ContentBudgeter(config)
    res = budgeter.trim_text(text)
    return res.text, res



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

