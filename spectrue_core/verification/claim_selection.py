"""Deterministic helpers for selecting the UI-facing main claim."""
from __future__ import annotations

from typing import Iterable

from spectrue_core.verification.calibration_registry import CalibrationRegistry
from spectrue_core.verification.claim_utility import score_claim_utility


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    role = (claim.get("claim_role") or claim.get("role") or "").lower()
    harm = _safe_float(claim.get("harm_potential", 0.0))
    target = (claim.get("verification_target") or "").lower()

    if target in {"instruction", "procedure"}:
        return False
    if role == "background" and harm < 3.0:
        return False
    if harm >= 3.0:
        return True
    return role in {"thesis", "support"}


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
) -> tuple[float, int, float, int]:
    bucket = ui_bucket(str(claim.get("claim_role") or claim.get("role") or ""))
    score = ui_score(
        claim,
        calibration_registry=calibration_registry,
        centrality_map=centrality_map,
        max_pos=max_pos,
    )
    pos_norm = _position_norm(claim, max_pos=max_pos)
    pos = ui_position(claim)
    return (-score, bucket, pos_norm, pos)


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
    harm_priority = [
        c
        for c in admissible
        if _safe_float(c.get("harm_potential"), 0.0) >= 4.0
        and _safe_float(c.get("check_worthiness"), 0.0) >= 0.5
        and ui_bucket(str(c.get("claim_role") or c.get("role") or "")) <= 1
    ]
    candidate_pool = harm_priority if harm_priority else admissible
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
    # Fallback to any claim if none admissible; caller should log this.
    return min(
        claims,
        key=lambda c: ui_sort_key(
            c,
            calibration_registry=calibration_registry,
            centrality_map=centrality_map,
            max_pos=max_pos,
        ),
    )


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
