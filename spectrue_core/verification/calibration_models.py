# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Calibration helpers for learned scoring."""

from __future__ import annotations

import math
from typing import Mapping, Any


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def linear_score(
    features: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    bias: float = 0.0,
    clamp: bool = True,
) -> tuple[float, float]:
    raw = float(bias)
    for key, weight in (weights or {}).items():
        raw += float(weight) * _safe_float(features.get(key), default=0.0)
    score = raw
    if clamp:
        score = max(0.0, min(1.0, score))
    return raw, score


def logistic_score(
    features: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    bias: float = 0.0,
) -> tuple[float, float]:
    raw = float(bias)
    for key, weight in (weights or {}).items():
        raw += float(weight) * _safe_float(features.get(key), default=0.0)
    return raw, _sigmoid(raw)


def pairwise_rank_score(
    features: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    bias: float = 0.0,
) -> tuple[float, float]:
    return logistic_score(features, weights, bias=bias)


def build_trace_payload(
    *,
    model_name: str,
    model_version: str,
    features: Mapping[str, Any],
    weights: Mapping[str, float],
    bias: float,
    raw_score: float,
    score: float,
    fallback_used: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contributions = {
        key: float(weights.get(key, 0.0)) * _safe_float(features.get(key), default=0.0)
        for key in (weights or {})
    }
    payload = {
        "model": model_name,
        "version": model_version,
        "bias": float(bias),
        "features": {k: _safe_float(v, default=0.0) for k, v in (features or {}).items()},
        "weights": dict(weights or {}),
        "contributions": contributions,
        "raw_score": float(raw_score),
        "score": float(score),
        "fallback_used": bool(fallback_used),
    }
    if extra:
        payload["extra"] = extra
    return payload
