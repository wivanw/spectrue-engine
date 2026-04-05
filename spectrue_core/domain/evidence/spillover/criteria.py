"""
Compatibility criteria for evidence spillover.
"""
from typing import Any
from .models import (
    ClaimEventSignature,
    EvidenceEventSignature,
    SignatureCompatible,
    SlotsFromAssertionKey,
    RequiredSlotsForTarget,
    MergeCovers,
)

def claim_assertion_keys(claim: dict[str, Any]) -> tuple[set[str], set[str]]:
    """
    Return (fact_keys, context_keys) from claim.assertions[] if present.
    Deterministic and schema-driven.
    """
    fact: set[str] = set()
    ctx: set[str] = set()
    assertions = claim.get("assertions")
    if not isinstance(assertions, list):
        return fact, ctx

    for a in assertions:
        if not isinstance(a, dict):
            continue
        key = a.get("key")
        if not key:
            continue
        dim = str(a.get("dimension") or "FACT").upper()
        if dim == "CONTEXT":
            ctx.add(str(key))
        else:
            fact.add(str(key))
    return fact, ctx


def is_transfer_candidate(src: dict[str, Any]) -> bool:
    """
    Conservative: transfer only evidence with an explainability anchor.
    No text heuristics.
    """
    stance = str(src.get("stance") or "").upper()
    if stance in {"IRRELEVANT"}:
        return False
    # Prevent cascade: transferred items must not be used as donors.
    if src.get("provenance") == "transferred":
        return False
    # Require an anchor for downstream explainability/judge
    if src.get("quote") or src.get("quote_span") or src.get("contradiction_span"):
        return True
    return False


# Claim-type compatibility for spillover: prevents e.g. numeric data
# spilling into attribution claims.  Derived from the 5-type taxonomy
# (core, numeric, timeline, attribution, sidefact) where each type has
# distinct evidence requirements.
_SPILLOVER_COMPATIBLE: dict[str, frozenset[str]] = {
    "core": frozenset({"core", "timeline", "sidefact"}),
    "timeline": frozenset({"timeline", "core"}),
    "numeric": frozenset({"numeric"}),
    "attribution": frozenset({"attribution"}),
    "sidefact": frozenset({"sidefact", "core"}),
}


def claim_type_compatible(
    origin_claim: dict[str, Any],
    target_claim: dict[str, Any],
) -> bool:
    """Return True if evidence from *origin_claim* may spill into *target_claim*."""
    origin_type = str(origin_claim.get("type") or origin_claim.get("claim_type") or "core").lower()
    target_type = str(target_claim.get("type") or target_claim.get("claim_type") or "core").lower()
    allowed = _SPILLOVER_COMPATIBLE.get(target_type)
    if allowed is None:
        return True  # unknown type — permissive fallback
    return origin_type in allowed


def compatible_for_claim(src: dict[str, Any], fact_keys: set[str], context_keys: set[str]) -> bool:
    """
    Deterministic compatibility using:
    - assertion_key (evidence -> which assertion it applies to)
    - stance class (SUPPORT/REFUTE vs CONTEXT/MENTION)
    - claim assertions dimension (FACT/CONTEXT)
    """
    akey = str(src.get("assertion_key") or "")
    stance = str(src.get("stance") or "").upper()

    # Legacy whole-claim evidence: allow (routing v2 keeps it conservative elsewhere)
    if not akey:
        return True

    if stance in {"SUPPORT", "REFUTE", "MIXED"}:
        return akey in fact_keys

    if stance in {"CONTEXT", "MENTION"}:
        return akey in context_keys

    return False


def _extract_verification_target(claim: dict[str, Any]) -> str:
    """
    Extract verification_target from claim metadata (schema-driven).
    Handles both flattened and nested representations.
    """
    md = claim.get("metadata")
    if isinstance(md, dict) and md.get("verification_target"):
        return str(md.get("verification_target"))
    if claim.get("verification_target"):
        return str(claim.get("verification_target"))
    return ""


def covers_ok_for_claim(
    src: dict[str, Any],
    claim: dict[str, Any],
    slots_from_assertion_key: SlotsFromAssertionKey,
    required_slots_for_verification_target: RequiredSlotsForTarget,
    merge_covers: MergeCovers,
) -> bool:
    """
    Deterministic compatibility using slots:
    required_slots(verification_target) must intersect evidence covers.
    """
    vt = _extract_verification_target(claim)
    required = required_slots_for_verification_target(vt)
    if not required:
        return True

    covers = set(src.get("covers", []) or [])
    akey = str(src.get("assertion_key") or "")

    derived = slots_from_assertion_key(akey)
    merged = merge_covers(covers, derived)

    return bool(merged & required)


def event_ok_for_claim(
    src: dict[str, Any],
    claim: dict[str, Any],
    claim_event_signature: ClaimEventSignature,
    evidence_event_signature: EvidenceEventSignature,
    signature_compatible: SignatureCompatible,
) -> bool:
    """
    Deterministic gate using event signatures.
    """
    c_sig = claim_event_signature(claim)
    e_sig = evidence_event_signature(src)

    # Relaxed: if either has no signature, assume compatible for spillover
    if not c_sig or not e_sig:
        return True

    return signature_compatible(c_sig, e_sig)
