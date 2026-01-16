# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from __future__ import annotations

from typing import Iterable, Optional


def _norm_tag(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # Deterministic normalization: whitespace -> underscore, keep ascii-ish
    s = "_".join(s.split())
    return s[:64]


def build_topic_tags(
    *,
    topic_group: Optional[str] = None,
    topic_key: Optional[str] = None,
    subject_entities: Optional[Iterable[str]] = None,
    retrieval_seed_terms: Optional[Iterable[str]] = None,
    verification_target: Optional[str] = None,
    claim_role: Optional[str] = None,
    locale_signals: Optional[list[dict]] = None,
    time_signals: Optional[list[dict]] = None,
) -> list[str]:
    """
    Deterministic topic tags for routing / clustering.

    This is NOT semantic understanding. It's a stable categorization layer
    based on already extracted structured fields.
    """
    tags: list[str] = []

    tg = _norm_tag(topic_group or "")
    tk = _norm_tag(topic_key or "")
    if tg:
        tags.append(f"topic_group:{tg}")
    if tk:
        tags.append(f"topic_key:{tk}")

    if verification_target:
        tags.append(f"target:{_norm_tag(verification_target)}")
    if claim_role:
        tags.append(f"role:{_norm_tag(claim_role)}")

    # Locale/time buckets are useful for routing without "text heuristics"
    if isinstance(locale_signals, list) and locale_signals:
        # Use first signal if present
        s0 = locale_signals[0]
        loc = _norm_tag(str(s0.get("country") or s0.get("locale") or s0.get("city") or ""))
        if loc:
            tags.append(f"locale:{loc}")
            
    if isinstance(time_signals, list) and time_signals:
        s0 = time_signals[0]
        # Use coarse bucket if present
        tb = _norm_tag(str(s0.get("time_bucket") or s0.get("year") or s0.get("value") or ""))
        if tb:
            tags.append(f"time:{tb}")

    # Entities / seed terms (bounded, stable)
    # We intentionally cap to avoid tag explosion.
    def add_list(prefix: str, items: Optional[Iterable[str]], cap: int) -> None:
        if not items:
            return
        n = 0
        for x in items:
            if n >= cap:
                break
            t = _norm_tag(str(x))
            if not t:
                continue
            tags.append(f"{prefix}:{t}")
            n += 1

    add_list("ent", subject_entities, cap=5)
    add_list("seed", retrieval_seed_terms, cap=5)

    # Ensure deterministic order and uniqueness
    seen = set()
    out = []
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out
