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
"""
URL Utilities

Small helpers shared across tools. These functions must be side-effect free.
"""

from __future__ import annotations

from urllib.parse import urlparse


def normalize_host(host: str) -> str:
    host = (host or "").strip().lower()
    return host[4:] if host.startswith("www.") else host


def is_valid_public_http_url(url: str) -> bool:
    try:
        if not url:
            return False
        u = urlparse(str(url).strip())
        if u.scheme not in ("http", "https"):
            return False
        host = normalize_host(u.hostname or "")
        if host in ("127.0.0.1", "localhost"):
            return False
        return True
    except Exception:
        return False


def clean_url_for_cache(url: str) -> str:
    """
    Strip tracking query params/fragments for cache key reuse.
    """
    try:
        u = urlparse(url)
        q = ""
        if u.query:
            ignored = {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "fbclid",
                "gclid",
            }
            pairs = u.query.split("&")
            valid_pairs = [p for p in pairs if p.split("=")[0].lower() not in ignored]
            q = "&".join(sorted(valid_pairs))

        qs = f"?{q}" if q else ""
        return f"{u.scheme}://{u.netloc}{u.path}{qs}"
    except Exception:
        return url


def canonical_url_for_dedupe(url: str) -> str:
    """
    Canonical URL representation for deduplication (host+path, no query/fragment).
    """
    try:
        u = urlparse(url)
        return normalize_host(u.netloc or "") + (u.path or "").rstrip("/")
    except Exception:
        return (url or "").strip()

