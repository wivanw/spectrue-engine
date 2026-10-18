# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Global veracity (G) aggregation over per-claim scores (standard mode).

Policy (docs/VERIFICATION_METRICS.md §7.0):
- Claims that are not verifiable by nature (`verification_target = none`:
  predictions, opinions, ...) are NOT part of global G. They carry the typed
  status `UNVERIFIABLE_BY_NATURE` and are excluded from the anchor choice, the
  numerator and the denominator.
- Among the remaining claims: if the anchor (highest importance) is unverified,
  global G is -1.0 (unverified). Otherwise G is the importance-weighted mean, and
  an unverified factual claim contributes 0.0 while staying in the denominator
  (missing evidence reduces the global score; it is never ignored).
- If no claim is verifiable by nature, global G is -1.0 with that status.
- The text asserts its main claims jointly, so G is never above the least-true resolved main
  claim: P(A and B) <= min(P(A), P(B)). Context claims (explain-only roles: context, background,
  meta) are excluded from global G like unverifiable ones (owner, 2026-10-18).
"""

from __future__ import annotations

from typing import Any, Mapping

from spectrue_core.domain.claims.model.enums import VerificationTarget
from spectrue_core.domain.verification.verdict.rgba import RGBAStatus


def is_unverifiable_by_nature(claim: Mapping[str, Any]) -> bool:
    """True when the claim's verification_target is NONE (top-level or in metadata)."""
    target = claim.get("verification_target")
    if target is None:
        meta = claim.get("metadata")
        target = meta.get("verification_target") if isinstance(meta, Mapping) else getattr(meta, "verification_target", None)
    if target is None:  # unknown target is not "none": the claim stays in global G
        return False
    value = target.value if isinstance(target, VerificationTarget) else target
    return str(value).lower() == VerificationTarget.NONE.value


CONTEXT_ROLES = frozenset({"context", "background", "meta"})  # explain-only (RGBA role weight 0)


def is_context_claim(claim: Mapping[str, Any]) -> bool:
    role = claim.get("claim_role") or claim.get("role")
    if role is None:
        meta = claim.get("metadata")
        role = meta.get("claim_role") if isinstance(meta, Mapping) else getattr(meta, "claim_role", None)
    value = getattr(role, "value", role)
    return str(value or "").lower() in CONTEXT_ROLES


def aggregate_global_veracity(
    score_by_id: Mapping[str, float],
    importance_by_id: Mapping[str, float],
    unverifiable_ids: set[str],
    context_ids: set[str] = frozenset(),
) -> tuple[float, dict[str, Any]]:
    """(global G, policy record). Scores < 0 mean "unverified"."""
    out_ids = set(unverifiable_ids) | set(context_ids)
    eligible = {cid: imp for cid, imp in importance_by_id.items() if cid not in out_ids}
    excluded = sorted(cid for cid in importance_by_id if cid in unverifiable_ids)
    excluded_context = sorted(cid for cid in importance_by_id if cid in context_ids and cid not in unverifiable_ids)
    if not eligible and excluded_context:
        # A text made only of context claims is judged on them rather than reported as unverifiable.
        eligible = {cid: importance_by_id[cid] for cid in excluded_context}
        excluded_context = []
    if not eligible:
        return -1.0, {
            "policy": "no_verifiable_claims",
            "status": RGBAStatus.UNVERIFIABLE_BY_NATURE.value,
            "excluded_unverifiable_by_nature": excluded,
        }

    anchor_id = max(eligible.items(), key=lambda kv: kv[1])[0]
    if float(score_by_id.get(anchor_id, -1.0)) < 0:
        return -1.0, {
            "policy": "anchor_unverified_blocks_global",
            "anchor_claim_id": anchor_id,
            "excluded_context": excluded_context,
            "excluded_unverifiable_by_nature": excluded,
        }

    num = den = 0.0
    for cid, imp in eligible.items():
        w = max(0.0, min(1.0, float(imp)))
        g = float(score_by_id.get(cid, -1.0))
        num += w * (g if g >= 0 else 0.0)
        den += w
    mean = (num / den) if den > 0 else -1.0
    resolved = [float(score_by_id[c]) for c in eligible if float(score_by_id.get(c, -1.0)) >= 0]
    bound = min(resolved) if resolved else mean
    return min(mean, bound), {
        "policy": "importance_weighted_mean_capped_by_least_true",
        "least_true": round(bound, 4),
        "excluded_context": excluded_context,
        "anchor_claim_id": anchor_id,
        "excluded_unverifiable_by_nature": excluded,
    }
