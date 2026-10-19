# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Judge output consistency: the verdict label and the score are two readings of one judgment.

A label that says the claim is true with a score below 0.5, or a label that says it is false with
a score above 0.5, is a contradiction (production 2026-10-18: "refuted by NASA", score 0.99). The
judge is re-asked once with the contradiction named; if it still contradicts itself the claim is
unverified (-1), never one of the two readings picked at random.
"""

from __future__ import annotations

from typing import Any

SUPPORT_LABELS = frozenset({"verified", "supported", "support", "partially_verified", "plausible", "likely",
                            "true", "mostly_true"})
REFUTE_LABELS = frozenset({"refuted", "refute", "unlikely", "false", "mostly_false", "debunked"})


def label_score_contradiction(label: Any, score: Any) -> str | None:
    """A short description of the contradiction, or None when label and score agree (or say 'unknown')."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s < 0:
        return None  # -1 = unverified: no direction to contradict
    lab = str(label or "").strip().lower()
    if lab in SUPPORT_LABELS and s < 0.5:
        return f"verdict '{lab}' says the claim is true but the score {s:.2f} says it is false"
    if lab in REFUTE_LABELS and s > 0.5:
        return f"verdict '{lab}' says the claim is false but the score {s:.2f} says it is true"
    return None


def contradictions(claim_verdicts: list[dict]) -> dict[str, str]:
    """claim_id -> contradiction, for a batch judge response (raw, before normalisation)."""
    out: dict[str, str] = {}
    for cv in claim_verdicts or []:
        if isinstance(cv, dict):
            why = label_score_contradiction(cv.get("verdict"), cv.get("verdict_score"))
            if why:
                out[str(cv.get("claim_id"))] = why
    return out


def retry_note(found: dict[str, str]) -> str:
    lines = "\n".join(f"- claim {cid}: {why}" for cid, why in found.items())
    return ("\n\n# CONSISTENCY CHECK\nYour previous answer contradicted itself:\n" + lines +
            "\nRe-evaluate. The verdict, verdict_score, rgba G and the reason must all state the same judgment.")


def mark_inconsistent(cv: dict, why: str) -> None:
    """The judge contradicted itself twice: the claim is unverified, with the reason recorded."""
    cv["verdict_score"] = -1.0
    cv["verdict"] = "unverified"
    cv["verdict_state"] = "insufficient_evidence"
    cv["status"] = "judge_inconsistent"
    cv["judge_inconsistency"] = why
    rgba = cv.get("rgba")
    if isinstance(rgba, list) and len(rgba) >= 2:
        cv["rgba"] = [rgba[0], -1.0, *rgba[2:]]
