# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.schema import (
    AssertionVerdict,
    ClaimVerdict,
    StructuredDebug,
    StructuredVerdict,
    VerdictStatus,
    VerdictState,
)
from spectrue_core.agents.skills.scoring_sanitization import maybe_drop_style_section, strip_internal_source_markers
from spectrue_core.utils.trace import Trace
import logging

logger = logging.getLogger(__name__)


def safe_score(val, default: float = -1.0) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default

    if default < 0 and (f < 0.0 or f > 1.0):
        return default
    return max(0.0, min(1.0, f))


def clamp_score_evidence_result(result: dict, *, judge_mode: str = "standard") -> dict:
    """
    Normalize/validate LLM scoring result.
    
    Args:
        result: Raw LLM output dict
        judge_mode: "standard" or "deep"
            - "standard": Apply normalization, defaults, thresholds
            - "deep": Minimal validation only, preserve LLM output 1:1
    """
    # Deep mode: minimal validation, no defaults/normalization
    is_deep = judge_mode == "deep"

    Trace.event("score_evidence.parsed_keys", {"keys": sorted(result.keys()), "judge_mode": judge_mode})

    if "verified_score" in result:
        result["verified_score"] = safe_score(result["verified_score"], default=-1.0)
    else:
        # Do NOT synthesize article-level G here (e.g., averaging claims).
        # Pipeline selects article-level G from a single claim (anchor/thesis).
        result["verified_score"] = -1.0

    explainability_raw = result.get("explainability_score")
    confidence_raw = result.get("confidence_score")
    if explainability_raw is None and confidence_raw is None:
        raise ValueError("missing_explainability_score")
    if explainability_raw is None:
        explainability_raw = confidence_raw
    if confidence_raw is None:
        confidence_raw = explainability_raw

    result["explainability_score"] = safe_score(explainability_raw, default=-1.0)
    result["confidence_score"] = safe_score(confidence_raw, default=-1.0)
    if result["explainability_score"] < 0 or result["confidence_score"] < 0:
        raise ValueError("invalid_explainability_score")

    Trace.event(
        "score_evidence.parsed_scores",
        {
            "verified": result.get("verified_score"),
            "explainability": result.get("explainability_score"),
            "confidence": result.get("confidence_score"),
            "danger": result.get("danger_score"),
            "style": result.get("style_score"),
        },
    )

    result["danger_score"] = safe_score(result.get("danger_score"), default=-1.0)
    result["style_score"] = safe_score(result.get("style_score"), default=-1.0)

    # Canonical verdict sets
    CANONICAL_VERDICTS = {"verified", "refuted", "ambiguous"}
    CANONICAL_STATES = {"supported", "refuted", "conflicted", "insufficient_evidence"}

    claim_verdicts = result.get("claim_verdicts")
    if isinstance(claim_verdicts, list):
        for cv in claim_verdicts:
            if isinstance(cv, dict):
                cid = cv.get("claim_id")

                # Deep mode: skip normalization for claims with status=error
                if is_deep and cv.get("status") == "error":
                    Trace.event(
                        "verdict.deep_error_claim_skipped",
                        {"claim_id": cid, "error_type": cv.get("error_type")},
                    )
                    continue

                # Track schema issues for this claim
                claim_schema_issues: dict[str, str] = {}

                # Deep mode: preserve LLM verdict_score without default
                if is_deep:
                    raw_vs = cv.get("verdict_score")
                    if raw_vs is None:
                        claim_schema_issues["verdict_score"] = "missing"
                    elif not isinstance(raw_vs, (int, float)):
                        claim_schema_issues["verdict_score"] = f"invalid type: {type(raw_vs).__name__}"
                    else:
                        p = safe_score(raw_vs, default=-1.0)
                        if p >= 0:
                            cv["verdict_score"] = p
                        else:
                            claim_schema_issues["verdict_score"] = f"out of range: {raw_vs}"
                else:
                    # Standard mode: allow -1.0 default
                    raw_vs = cv.get("verdict_score")
                    if raw_vs is None:
                        claim_schema_issues["verdict_score"] = "missing, using default -1.0"
                    
                    # Use -1.0 as default if missing
                    # Note: safe_score allows -1.0 if default is -1.0
                    p = safe_score(cv.get("verdict_score", -1.0), default=-1.0)
                    cv["verdict_score"] = p

                # Log schema issues for this claim
                if claim_schema_issues:
                    logger.warning(
                        "[Scoring] Claim %s schema issues: %s",
                        cid, claim_schema_issues,
                    )
                    Trace.event(
                        "verdict.claim_schema_mismatch",
                        {
                            "claim_id": cid,
                            "judge_mode": judge_mode,
                            "issues": claim_schema_issues,
                            "received_keys": list(cv.keys()),
                        },
                    )

                # p is already set above
                # p = cv.get("verdict_score", 0.5) if not is_deep else cv.get("verdict_score", -1.0)

                # --- (1) Normalize verdict_state from p ---
                # Deep mode: skip this normalization, preserve LLM output
                if not is_deep:
                    # Contract: p >= 0.6 → supported, p <= 0.4 → refuted
                    # p < 0 -> unverified (insufficient)
                    raw_state = str(cv.get("verdict_state") or "").lower().strip()
                    if raw_state not in CANONICAL_STATES:
                        # Derive verdict_state from p
                        if p < 0:
                            cv["verdict_state"] = "insufficient_evidence"
                        elif p >= 0.6:
                            cv["verdict_state"] = "supported"
                        elif p <= 0.4:
                            cv["verdict_state"] = "refuted"
                        else:
                            # 0.4 < p < 0.6 → conflicted (evidence exists but ambiguous)
                            cv["verdict_state"] = "conflicted"
                        
                        Trace.event(
                            "verdict.state_normalized",
                            {
                                "claim_id": cv.get("claim_id"),
                                "p": p,
                                "raw_state": raw_state or None,
                                "normalized_state": cv["verdict_state"],
                            },
                        )

                # --- (2) Normalize verdict (UI label) from p ---
                # Deep mode: skip this normalization, preserve LLM output
                raw_verdict = str(cv.get("verdict") or "").lower().strip()
                if not is_deep and raw_verdict not in CANONICAL_VERDICTS:
                    # Log non-canonical LLM verdict
                    Trace.event(
                        "verdict.label_noncanonical",
                        {
                            "claim_id": cv.get("claim_id"),
                            "raw_verdict": raw_verdict or None,
                            "p": p,
                        },
                    )
                    # Normalize to canonical UI label
                    if p < 0:
                        cv["verdict"] = "unverified"
                    elif p >= 0.8:
                        cv["verdict"] = "verified"
                    elif p <= 0.2:
                        cv["verdict"] = "refuted"
                    else:
                        cv["verdict"] = "ambiguous"

                # --- (3) Semantic contract: insufficient evidence = -1.0 (was 0.5) ---
                # Deep mode: skip this normalization, preserve LLM output
                if not is_deep:
                    vs = str(cv.get("verdict_state") or "").lower().strip()
                    if vs in {"insufficient_evidence", "insufficient evidence", "insufficient", "unverified"}:
                        if cv["verdict_score"] != -1.0:
                            cv["verdict_score"] = -1.0

                # --- (4) Validate and normalize per-claim RGBA ---
                # Deep mode: if RGBA is None (error claim), don't touch it
                raw_rgba = cv.get("rgba")
                if raw_rgba is None and is_deep:
                    # Deep mode error claim - leave rgba as None
                    pass
                elif isinstance(raw_rgba, list) and len(raw_rgba) == 4:
                    try:
                        # Clamp all values to [0, 1]
                        normalized_rgba = [
                            max(0.0, min(1.0, float(raw_rgba[0]))),  # R (danger)
                            max(0.0, min(1.0, float(raw_rgba[1]))),  # G (veracity)
                            max(0.0, min(1.0, float(raw_rgba[2]))),  # B (style)
                            max(0.0, min(1.0, float(raw_rgba[3]))),  # A (explainability)
                        ]
                        cv["rgba"] = normalized_rgba
                        Trace.event(
                            "verdict.rgba_parsed",
                            {
                                "claim_id": cv.get("claim_id"),
                                "rgba": normalized_rgba,
                            },
                        )
                    except (TypeError, ValueError, IndexError):
                        # Invalid rgba format
                        if is_deep:
                            # Deep mode: trace error, keep rgba as None (no fallback)
                            cv["rgba"] = None
                            Trace.event(
                                "verdict.rgba_invalid_deep",
                                {
                                    "claim_id": cv.get("claim_id"),
                                    "raw_rgba": raw_rgba,
                                    "error": "Invalid RGBA format in deep mode - no fallback",
                                },
                            )
                        else:
                            # Standard mode: remove it so fallback will be used
                            cv.pop("rgba", None)
                            Trace.event(
                                "verdict.rgba_invalid",
                                {
                                    "claim_id": cv.get("claim_id"),
                                    "raw_rgba": raw_rgba,
                                },
                            )
                else:
                    # No rgba or wrong format
                    if is_deep:
                        # Deep mode: keep as None (no fallback)
                        pass
                    else:
                        # Standard mode: fallback will be used later
                        pass
    else:
        result["claim_verdicts"] = []

    return result


def parse_structured_verdict(raw: dict, *, lang: str = "en") -> StructuredVerdict:
    def parse_status(s: str) -> VerdictStatus:
        s_lower = (s or "").lower()
        if s_lower in ("verified", "confirmed"):
            return VerdictStatus.VERIFIED
        if s_lower in ("refuted", "false"):
            return VerdictStatus.REFUTED
        if s_lower == "partially_verified":
            return VerdictStatus.PARTIALLY_VERIFIED
        if s_lower == "unverified":
            return VerdictStatus.UNVERIFIED
        return VerdictStatus.AMBIGUOUS

    def parse_state(s: str | None) -> VerdictState:
        s_lower = (s or "").lower().strip()
        if s_lower in ("supported", "support"):
            return VerdictState.SUPPORTED
        if s_lower in ("refuted", "refute"):
            return VerdictState.REFUTED
        if s_lower in ("conflicted", "conflict"):
            return VerdictState.CONFLICTED
        if s_lower in ("insufficient_evidence", "insufficient evidence", "insufficient"):
            return VerdictState.INSUFFICIENT_EVIDENCE
        return VerdictState.INSUFFICIENT_EVIDENCE

    claim_verdicts: list[ClaimVerdict] = []
    raw_claims = raw.get("claim_verdicts", [])

    if isinstance(raw_claims, list):
        for rc in raw_claims:
            if not isinstance(rc, dict):
                continue

            raw_conf = rc.get("confidence")
            if isinstance(raw_conf, str):
                confidence_label = raw_conf.lower().strip()
            elif isinstance(rc.get("confidence_label"), str):
                confidence_label = str(rc.get("confidence_label")).lower().strip()
            else:
                confidence_label = "low"
            if confidence_label not in ("low", "medium", "high"):
                confidence_label = "low"

            reasons_short = rc.get("reasons_short", [])
            if not isinstance(reasons_short, list):
                reasons_short = []

            reasons_expert = rc.get("reasons_expert", {})
            if not isinstance(reasons_expert, dict):
                reasons_expert = {}

            assertion_verdicts: list[AssertionVerdict] = []
            raw_assertions = rc.get("assertion_verdicts", [])

            fact_verified = 0
            fact_total = 0

            if isinstance(raw_assertions, list):
                for ra in raw_assertions:
                    if not isinstance(ra, dict):
                        continue

                    dim = (ra.get("dimension") or "FACT").upper()
                    if dim == "FACT":
                        fact_total += 1
                        score = safe_score(ra.get("score"), default=-1.0)
                        if score >= 0.6:
                            fact_verified += 1

                    assertion_verdicts.append(
                        AssertionVerdict(
                            assertion_key=ra.get("assertion_key", ""),
                            dimension=dim,
                            status=parse_status(ra.get("status", "")),
                            score=safe_score(ra.get("score"), default=-1.0),
                            evidence_count=int(ra.get("evidence_count", 0)),
                            supporting_urls=ra.get("supporting_urls", []),
                            rationale=ra.get("rationale", ""),
                        )
                    )

            claim_verdicts.append(
                ClaimVerdict(
                    claim_id=rc.get("claim_id", ""),
                    status=parse_status(rc.get("status", "")),
                    verdict=parse_status(rc.get("verdict", rc.get("status", ""))),
                    verdict_state=parse_state(rc.get("verdict_state")),
                    verdict_score=0.0,  # temp, set below
                    assertion_verdicts=assertion_verdicts,
                    evidence_count=int(rc.get("evidence_count", 0)),
                    fact_assertions_verified=fact_verified,
                    fact_assertions_total=fact_total,
                    reason=rc.get("reason", ""),
                    key_evidence=rc.get("key_evidence", []),
                    confidence=confidence_label,
                    reasons_short=reasons_short,
                    reasons_expert=reasons_expert,
                )
            )
            # Enforce unverified score for insufficient evidence
            if claim_verdicts[-1].verdict_state == VerdictState.INSUFFICIENT_EVIDENCE:
                original_score = safe_score(rc.get("verdict_score"), default=-1.0)
                if original_score != -1.0:
                    Trace.event(
                        "verdict.score_corrected",
                        {
                            "claim_id": claim_verdicts[-1].claim_id,
                            "from": original_score,
                            "to": -1.0,
                            "reason": "insufficient_evidence",
                        },
                    )
                claim_verdicts[-1].verdict_score = -1.0
            else:
                claim_verdicts[-1].verdict_score = safe_score(rc.get("verdict_score"), default=-1.0)

    verified = safe_score(raw.get("verified_score"), default=-1.0)
    explainability = safe_score(raw.get("explainability_score"), default=-1.0)
    danger = safe_score(raw.get("danger_score"), default=-1.0)
    style = safe_score(raw.get("style_score"), default=-1.0)

    # Do not average claim scores here; article-level G is selected in pipeline.
    if verified < 0:
        verified = -1.0

    rationale = strip_internal_source_markers(str(raw.get("rationale", "")))
    rationale = maybe_drop_style_section(rationale, honesty_score=style, lang=lang)

    debug = StructuredDebug(
        per_claim={cv.claim_id: {"score": cv.verdict_score, "status": cv.status.value} for cv in claim_verdicts},
        content_unavailable_count=0,
    )

    return StructuredVerdict(
        claim_verdicts=claim_verdicts,
        verified_score=verified,
        explainability_score=explainability,
        danger_score=danger,
        style_score=style,
        rationale=rationale,
        structured_debug=debug,
        overall_confidence=verified,
    )
