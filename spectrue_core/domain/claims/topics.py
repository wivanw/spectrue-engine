# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Claim topics derived from shared subject entities (M146 T013).

The LLM never produced `topic_group` / `topic_key`, so every claim was topic
"Other" and topic-based gating (spillover, edge typing) treated unrelated claims
as one topic. A topic is now a connected component of claims that share at least
one normalized subject entity: deterministic, no thresholds.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable


def normalize_entities(raw: Any) -> frozenset[str]:
    """Case- and whitespace-normalized subject entities (single definition)."""
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(" ".join(str(e).casefold().split()) for e in raw if str(e).strip())


def assign_topics(claims: Iterable[dict[str, Any]]) -> None:
    """Set `topic_key` and `topic_group` on each claim in place.

    Claims without entities each form their own topic (keyed by claim id).
    `topic_group` is the most frequent entity of the component (ties: alphabetical).
    """
    items = [c for c in claims if isinstance(c, dict)]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_owner: dict[str, int] = {}
    ents = [normalize_entities(c.get("subject_entities")) for c in items]
    for i, es in enumerate(ents):
        for e in es:
            if e in first_owner:
                parent[find(i)] = find(first_owner[e])
            else:
                first_owner[e] = i

    components: dict[int, list[int]] = {}
    for i in range(len(items)):
        components.setdefault(find(i), []).append(i)

    for members in components.values():
        counts: Counter[str] = Counter(e for i in members for e in ents[i])
        if counts:
            group = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            key = "tp_" + hashlib.sha256("|".join(sorted(counts)).encode()).hexdigest()[:12]
        for i in members:
            if counts:
                items[i]["topic_group"] = group
                items[i]["topic_key"] = key
            else:
                cid = str(items[i].get("id") or items[i].get("claim_id") or i)
                items[i]["topic_group"] = "Other"
                items[i]["topic_key"] = f"tp_claim_{cid}"
