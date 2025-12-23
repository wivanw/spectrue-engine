from spectrue_core.schema import (
    AssertionVerdict,
    ClaimVerdict,
    StructuredDebug,
    StructuredVerdict,
    VerdictStatus,
    VerdictState,
)
from spectrue_core.agents.skills.scoring_sanitization import maybe_drop_style_section, strip_internal_source_markers


def safe_score(val, default: float = -1.0) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default

    if default < 0 and (f < 0.0 or f > 1.0):
        return default
    return max(0.0, min(1.0, f))


def clamp_score_evidence_result(result: dict) -> dict:
    if "verified_score" in result:
        result["verified_score"] = safe_score(result["verified_score"], default=-1.0)
    else:
        verdicts = result.get("claim_verdicts")
        if isinstance(verdicts, list) and verdicts:
            vals: list[float] = []
            for v in verdicts:
                if isinstance(v, dict):
                    vals.append(safe_score(v.get("verdict_score", 0.5)))
            result["verified_score"] = (sum(vals) / len(vals)) if vals else -1.0
        else:
            result["verified_score"] = -1.0

    result["explainability_score"] = safe_score(result.get("explainability_score"), default=-1.0)
    result["danger_score"] = safe_score(result.get("danger_score"), default=-1.0)
    result["style_score"] = safe_score(result.get("style_score"), default=-1.0)

    claim_verdicts = result.get("claim_verdicts")
    if isinstance(claim_verdicts, list):
        for cv in claim_verdicts:
            if isinstance(cv, dict):
                cv["verdict_score"] = safe_score(cv.get("verdict_score", 0.5))
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
                    verdict_score=safe_score(rc.get("verdict_score"), default=-1.0),
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

    verified = safe_score(raw.get("verified_score"), default=-1.0)
    explainability = safe_score(raw.get("explainability_score"), default=-1.0)
    danger = safe_score(raw.get("danger_score"), default=-1.0)
    style = safe_score(raw.get("style_score"), default=-1.0)

    if verified < 0 and claim_verdicts:
        scores = [(cv.verdict_score, 1.0) for cv in claim_verdicts if cv.verdict_score >= 0]
        if scores:
            total_weight = sum(w for _, w in scores)
            if total_weight > 0:
                verified = sum(s * w for s, w in scores) / total_weight

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
