# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from spectrue_core.domain.claims.metadata import (
    ARTICLE_INTENTS,
    CLAIM_TYPE_MAPPING,
    DOMAIN_MAPPING,
    TOPIC_GROUPS,
    clamp_float,
    clamp_int,
    normalize_article_intent,
    normalize_claim_category,
    normalize_topic_group,
)

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


__all__ = [
    "ARTICLE_INTENTS",
    "TOPIC_GROUPS",
    "SEARCH_INTENTS",
    "DOMAIN_MAPPING",
    "CLAIM_TYPE_MAPPING",
    "clamp_float",
    "clamp_int",
    "normalize_topic_group",
    "normalize_article_intent",
    "normalize_claim_category",
]
