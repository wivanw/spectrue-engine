# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from spectrue_core.schema import ClaimDomain, ClaimType


TOPIC_GROUPS = [
    "Politics",
    "Economy",
    "War",
    "Science",
    "Technology",
    "Health",
    "Environment",
    "Society",
    "Sports",
    "Culture",
    "Other",
]

SEARCH_INTENTS = [
    "scientific_fact",
    "official_statement",
    "breaking_news",
    "historical_event",
    "quote_attribution",
    "prediction_opinion",
    "viral_rumor",
    "forecast",
]

ARTICLE_INTENTS = ["news", "evergreen", "official", "opinion", "prediction"]

DOMAIN_MAPPING = {
    "Politics": ClaimDomain.POLITICS,
    "Economy": ClaimDomain.FINANCE,
    "War": ClaimDomain.NEWS,
    "Science": ClaimDomain.SCIENCE,
    "Technology": ClaimDomain.TECHNOLOGY,
    "Health": ClaimDomain.HEALTH,
    "Environment": ClaimDomain.SCIENCE,
    "Society": ClaimDomain.NEWS,
    "Sports": ClaimDomain.SPORTS,
    "Culture": ClaimDomain.ENTERTAINMENT,
    "Other": ClaimDomain.OTHER,
}

CLAIM_TYPE_MAPPING = {
    "core": ClaimType.EVENT,
    "numeric": ClaimType.NUMERIC,
    "timeline": ClaimType.TIMELINE,
    "attribution": ClaimType.ATTRIBUTION,
    "sidefact": ClaimType.OTHER,
    "atomic": ClaimType.OTHER,
    "causal": ClaimType.OTHER,
    "comparative": ClaimType.COMPARISON,
    "policy_plan": ClaimType.POLICY,
    "definition": ClaimType.DEFINITION,
    "future": ClaimType.TIMELINE,
    "existence": ClaimType.OTHER,
}


def clamp_float(value, *, default: float, lo: float, hi: float) -> float:
    try:
        f = float(value)
    except Exception:
        f = float(default)
    return max(lo, min(hi, f))


def clamp_int(value, *, default: int, lo: int, hi: int) -> int:
    try:
        i = int(value)
    except Exception:
        i = int(default)
    return max(lo, min(hi, i))


def normalize_topic_group(topic: str | None, *, default: str = "Other") -> str:
    t = (topic or default) or default
    return t if t in TOPIC_GROUPS else default


def normalize_article_intent(intent: str | None, *, default: str = "news") -> str:
    v = (intent or default) or default
    return v if v in ARTICLE_INTENTS else default


def normalize_claim_category(category: str | None, *, default: str = "FACTUAL") -> str:
    v = (category or default) or default
    allowed = {"FACTUAL", "SATIRE", "OPINION", "HYPERBOLIC"}
    return v if v in allowed else default
