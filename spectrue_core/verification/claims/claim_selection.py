# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Deterministic helpers for selecting the UI-facing main claim."""
from __future__ import annotations

from typing import Iterable

from spectrue_core.utils.trace import Trace
from spectrue_core.verification.calibration.calibration_registry import CalibrationRegistry
from spectrue_core.verification.claims.claim_utility import score_claim_utility


def _has_nonempty_text(claim: dict) -> bool:
    text = claim.get("normalized_text") or claim.get("text") or claim.get("claim_text") or ""
    return bool(str(text).strip())


def _has_anchor_position(claim: dict) -> bool:
    anchor = claim.get("anchor")
    if not isinstance(anchor, dict):
        return False
    try:
        start = int(anchor.get("char_start"))
        end = int(anchor.get("char_end"))
    except (TypeError, ValueError):
        return False
    if start < 0 or end <= start:
        return False
    text = claim.get("normalized_text") or claim.get("text") or claim.get("claim_text") or ""
    if text:
        try:
            if end > len(str(text)):
                return False
        except Exception:
            # If text is not stringable, don't block selection solely on this check.
            pass
    return True


def ui_bucket(role: str | None) -> int:
    """Lower bucket = higher priority."""
    mapping = {"core": 0, "thesis": 0, "support": 1, "counter": 1, "background": 2}
    if not role:
        return 2
    try:
        role_str = str(role).strip().lower()
    except Exception:
        return 2
    return mapping.get(role_str, 2)


def ui_position(claim: dict) -> int:
    """Character start used as stable tie-breaker; later = larger value."""
    if not isinstance(claim, dict):
        return 10**9
    anchor = claim.get("anchor") or {}
    if not isinstance(anchor, dict):
        return 10**9
    try:
        return int(anchor.get("char_start", 10**9))
    except (TypeError, ValueError):
        return 10**9


def ui_score(
    claim: dict,
    *,
    calibration_registry: CalibrationRegistry | None = None,
    centrality_map: dict[str, float] | None = None,
    max_pos: int | None = None,
) -> float:
    if not isinstance(claim, dict):
        return 0.0
    score, _trace = score_claim_utility(
        claim,
        centrality_map=centrality_map,
        max_pos=max_pos,
        calibration_registry=calibration_registry,
    )
    return float(score)


def is_admissible_as_main(claim: dict) -> bool:
    if not isinstance(claim, dict):
        return False
    claim_id = claim.get("id") or claim.get("claim_id")
    if not claim_id:
        return False
    if not _has_nonempty_text(claim):
        return False
    if not _has_anchor_position(claim):
        return False
    target = (claim.get("verification_target") or "").lower()
    if target in {"instruction", "procedure"}:
        return False
    return True


def _fallback_main_claim(claims: list[dict]) -> dict | None:
    """
    Strict fallback when no claim satisfies admissibility contracts.
    We prefer: has id + nonempty text; anchor may be missing.
    """
    best: dict | None = None
    best_len = -1
    for claim in claims:
        claim_id = claim.get("id") or claim.get("claim_id")
        if not claim_id:
            continue
        text = claim.get("normalized_text") or claim.get("text") or claim.get("claim_text") or ""
        text_val = str(text).strip()
        if not text_val:
            continue
        if len(text_val) > best_len:
            best = claim
            best_len = len(text_val)
    return best


def _position_norm(claim: dict, *, max_pos: int | None) -> float:
    pos = ui_position(claim)
    if not max_pos or max_pos <= 0:
        return 1.0 if pos > 0 else 0.0
    return max(0.0, min(1.0, pos / max_pos))


def ui_sort_key(
    claim: dict,
    *,
    calibration_registry: CalibrationRegistry | None = None,
    centrality_map: dict[str, float] | None = None,
    max_pos: int | None = None,
) -> tuple[float, float, int]:
    score = ui_score(
        claim,
        calibration_registry=calibration_registry,
        centrality_map=centrality_map,
        max_pos=max_pos,
    )
    pos_norm = _position_norm(claim, max_pos=max_pos)
    pos = ui_position(claim)
    return (-score, pos_norm, pos)


def pick_ui_main_claim(
    claims: list[dict],
    *,
    calibration_registry: CalibrationRegistry | None = None,
    centrality_map: dict[str, float] | None = None,
) -> dict | None:
    if not claims:
        return None
    max_pos = max((ui_position(c) for c in claims if isinstance(c, dict)), default=0)
    admissible = [c for c in claims if is_admissible_as_main(c)]
    candidate_pool = admissible
    if candidate_pool:
        return min(
            candidate_pool,
            key=lambda c: ui_sort_key(
                c,
                calibration_registry=calibration_registry,
                centrality_map=centrality_map,
                max_pos=max_pos,
            ),
        )
    fallback = _fallback_main_claim(claims)
    Trace.event(
        "anchor_selection.no_admissible_claims",
        {
            "num_claims": len(claims),
            "num_admissible": 0,
            "fallback_used": bool(fallback),
            "fallback_claim_id": (fallback.get("id") or fallback.get("claim_id")) if fallback else None,
        },
    )
    return fallback


def top_ui_candidates(
    claims: Iterable[dict],
    *,
    limit: int = 3,
    calibration_registry: CalibrationRegistry | None = None,
    centrality_map: dict[str, float] | None = None,
) -> list[dict]:
    items = list(claims or [])
    if not items:
        return []
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 3
    max_pos = max((ui_position(c) for c in items if isinstance(c, dict)), default=0)
    return sorted(
        items,
        key=lambda c: ui_sort_key(
            c,
            calibration_registry=calibration_registry,
            centrality_map=centrality_map,
            max_pos=max_pos,
        ),
    )[:limit]
