# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from pathlib import Path

import diskcache


def ensure_diskcache(path: Path) -> diskcache.Cache:
    path.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(path))


def normalize_domains(domains: list[str] | None) -> list[str] | None:
    if not domains:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for domain in domains:
        if not domain:
            continue
        normalized = domain.lower().lstrip(".")
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out or None


def normalize_exclude_domains(domains: list[str] | None, *, cap: int = 32) -> list[str] | None:
    if not domains:
        return None
    normalized = sorted({d.lower().lstrip(".") for d in domains if d})
    if cap and len(normalized) > cap:
        normalized = normalized[:cap]
    return normalized or None


def make_search_cache_key(
    *,
    query: str,
    depth: str,
    limit: int,
    raw_mode: bool,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    topic: str,
) -> str:
    q_cache = (query or "").lower()
    include_key = ",".join(sorted(include_domains or []))[:2000] if include_domains else ""
    exclude_key = ",".join(exclude_domains or []) if exclude_domains else ""
    return f"{q_cache}|{depth}|{limit}|{int(bool(raw_mode))}|{include_key}|{exclude_key}|{topic}"


def get_cached_context_and_sources(cache: diskcache.Cache, key: str) -> tuple[str, list[dict]] | None:
    try:
        cached_data = cache[key]
    except Exception:
        return None

    if isinstance(cached_data, tuple) and len(cached_data) == 2:
        context_str, sources_list = cached_data
        if isinstance(context_str, str) and isinstance(sources_list, list):
            return context_str, sources_list
        return None

    # Old cache format (string only) — invalidate.
    try:
        del cache[key]
    except Exception:
        pass
    return None

