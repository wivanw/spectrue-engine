"""Deterministic helpers for selecting the UI-facing main claim."""
from __future__ import annotations

from typing import Iterable


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


def ui_score(claim: dict) -> float:
    if not isinstance(claim, dict):
        return 0.0
    worthiness = _safe_float(claim.get("check_worthiness", claim.get("importance", 0.0)))
    harm = _safe_float(claim.get("harm_potential", 0.0))
    importance = _safe_float(claim.get("importance", 0.0))
    return (2.0 * worthiness) + (1.5 * harm) + (0.5 * importance)


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


def ui_sort_key(claim: dict, *, max_pos: int | None = None) -> tuple[int, float, float, int]:
    bucket = ui_bucket(str(claim.get("claim_role") or claim.get("role") or ""))
    score = ui_score(claim)
    pos_norm = _position_norm(claim, max_pos=max_pos)
    # Small positional penalty inside the same bucket/score band.
    score_with_pos = score - 0.05 * pos_norm
    pos = ui_position(claim)
    return (bucket, -score_with_pos, pos_norm, pos)


def pick_ui_main_claim(claims: list[dict]) -> dict | None:
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
        return min(candidate_pool, key=lambda c: ui_sort_key(c, max_pos=max_pos))
    # Fallback to any claim if none admissible; caller should log this.
    return min(claims, key=lambda c: ui_sort_key(c, max_pos=max_pos))


def top_ui_candidates(claims: Iterable[dict], *, limit: int = 3) -> list[dict]:
    items = list(claims or [])
    if not items:
        return []
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 3
    max_pos = max((ui_position(c) for c in items if isinstance(c, dict)), default=0)
    return sorted(items, key=lambda c: ui_sort_key(c, max_pos=max_pos))[:limit]
