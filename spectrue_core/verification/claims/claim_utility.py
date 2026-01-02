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
"""Canonical claim utility scoring for anchor selection and ordering."""

from __future__ import annotations

from typing import Any

from spectrue_core.runtime_config import CalibrationPolicyConfig
from spectrue_core.verification.calibration.calibration_registry import CalibrationRegistry


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _role_weight(role: str | None, *, policy: CalibrationPolicyConfig) -> float:
    mapping = policy.claim_utility_role_weights or {}
    bg = float(mapping.get("background", 0.2))
    if not role:
        return bg
    return float(mapping.get(str(role).strip().lower(), bg))


def _claim_position(claim: dict) -> int:
    if not isinstance(claim, dict):
        return 10**9
    anchor = claim.get("anchor") or {}
    if not isinstance(anchor, dict):
        return 10**9
    try:
        return int(anchor.get("char_start", 10**9))
    except (TypeError, ValueError):
        return 10**9


def _position_norm(claim: dict, *, max_pos: int | None) -> float:
    pos = _claim_position(claim)
    if not max_pos or max_pos <= 0:
        return 1.0 if pos > 0 else 0.0
    return _clamp01(pos / max_pos)


def _lede_bonus(claim: dict, *, window: int) -> float:
    if not isinstance(claim, dict):
        return 0.0
    anchor = claim.get("anchor") or {}
    if not isinstance(anchor, dict):
        return 0.0
    try:
        pos = int(anchor.get("char_start"))
    except (TypeError, ValueError):
        return 0.0
    if pos < 0:
        return 0.0
    if pos >= window:
        return 0.0
    return _clamp01(1.0 - (pos / window))


def build_claim_utility_features(
    claim: dict,
    *,
    centrality_map: dict[str, float] | None = None,
    max_pos: int | None = None,
    calibration_registry: CalibrationRegistry | None = None,
) -> dict[str, float]:
    centrality_map = centrality_map or {}
    registry = calibration_registry or CalibrationRegistry.from_runtime(None)
    policy = registry.policy
    claim_id = str(claim.get("id") or claim.get("claim_id") or "")
    role = claim.get("claim_role") or claim.get("role")
    worthiness = _safe_float(claim.get("check_worthiness", claim.get("importance", 0.0)))
    harm_raw = claim.get("harm_potential", 0.0)
    harm = _safe_float(harm_raw, default=0.0)
    harm_scale = float(policy.claim_utility_harm_scale or 1.0)
    if harm > 1.0 and harm_scale > 0:
        harm = harm / harm_scale
    importance = _safe_float(claim.get("importance", 0.0))
    centrality = _safe_float(centrality_map.get(claim_id, claim.get("centrality", 0.0)))

    return {
        "role_weight": _clamp01(_role_weight(str(role) if role is not None else None, policy=policy)),
        "worthiness": _clamp01(worthiness),
        "harm": _clamp01(harm),
        "importance": _clamp01(importance),
        "centrality": _clamp01(centrality),
        "lede_bonus": _clamp01(
            _lede_bonus(claim, window=int(policy.claim_utility_lede_window))
        ),
        "position_norm": _position_norm(claim, max_pos=max_pos),
    }


def score_claim_utility(
    claim: dict,
    *,
    centrality_map: dict[str, float] | None = None,
    max_pos: int | None = None,
    calibration_registry: CalibrationRegistry | None = None,
) -> tuple[float, dict[str, Any]]:
    registry = calibration_registry or CalibrationRegistry.from_runtime(None)
    features = build_claim_utility_features(
        claim,
        centrality_map=centrality_map,
        max_pos=max_pos,
        calibration_registry=registry,
    )
    model = registry.get_model("claim_utility")
    if not model:
        raw_score = sum(features.values())
        return _clamp01(raw_score), {"model": "claim_utility", "fallback_used": True, "features": features}
    score, trace = model.score(features)
    if isinstance(trace, dict):
        trace.setdefault("extra", {})
        trace["extra"].update(
            {
                "claim_utility_lede_window": int(registry.policy.claim_utility_lede_window),
                "claim_utility_harm_scale": float(registry.policy.claim_utility_harm_scale),
                "claim_utility_role_weights": dict(registry.policy.claim_utility_role_weights or {}),
            }
        )
    return score, trace
