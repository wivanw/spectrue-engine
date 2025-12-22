# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Source Utilities (Canonical)

Normalization rules:
- Prefer `url` but accept provider-style `link`
- Prefer `content` but accept provider-style `snippet`

These helpers are intentionally small and side-effect free.
"""

from __future__ import annotations

from typing import Any, Iterable
from urllib.parse import urlparse

from spectrue_core.verification.types import Source


def extract_domain(url: str) -> str:
    """Best-effort domain extraction from a URL."""
    if not url:
        return ""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def canonicalize_source(src: Any) -> Source | None:
    """Normalize a single source dict to the canonical Source shape."""
    if not isinstance(src, dict):
        return None

    out: Source = dict(src)  # shallow copy

    if "url" not in out and out.get("link"):
        out["url"] = str(out.get("link") or "")
    if "content" not in out and out.get("snippet"):
        out["content"] = str(out.get("snippet") or "")

    url = str(out.get("url") or "")
    if url and "domain" not in out:
        out["domain"] = extract_domain(url)

    return out


def canonicalize_sources(sources: Iterable[Any]) -> list[Source]:
    """Normalize many sources and drop non-dict entries."""
    out: list[Source] = []
    for s in sources:
        cs = canonicalize_source(s)
        if cs is not None:
            out.append(cs)
    return out


def has_evidence_chunk(source: Any) -> bool:
    """
    Check whether a source includes a usable evidence chunk.

    Evidence chunk = quote/snippet/content (provider-dependent).
    """
    if not isinstance(source, dict):
        return False
    return bool(source.get("quote") or source.get("snippet") or source.get("content"))

