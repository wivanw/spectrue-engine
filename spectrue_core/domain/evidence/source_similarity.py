# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Single definition of a search result's claim-similarity signal (rerank, G1, G3)."""

from __future__ import annotations

from typing import Any


def unit_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(1.0, score))


def provider_score(src: dict[str, Any]) -> float:
    return unit_score(src.get("provider_score") if src.get("provider_score") is not None else src.get("score"))


def similarity_score(src: dict[str, Any]) -> float:
    """similarity_score -> sim -> calibrated relevance_score -> provider score."""
    sim = unit_score(src.get("similarity_score") if src.get("similarity_score") is not None else src.get("sim"))
    if sim == 0.0:
        sim = unit_score(src.get("relevance_score"))
    if sim == 0.0:
        sim = provider_score(src)
    return sim
