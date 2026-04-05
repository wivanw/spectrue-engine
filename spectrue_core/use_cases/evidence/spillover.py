"""Evidence spillover use cases."""

from __future__ import annotations

from typing import Any, Callable

from spectrue_core.domain.evidence.spillover import compute_spillover
from spectrue_core.domain.evidence.slot_maps import (
    merge_covers,
    required_slots_for_verification_target,
    slots_from_assertion_key,
)
from spectrue_core.domain.evidence.event_signature import (
    claim_event_signature,
    evidence_event_signature,
    signature_compatible,
)


def run_spillover(
    *,
    sources: list[dict],
    claims: list[dict],
    cluster_map: dict[str, str],
    top_k: int,
    normalize_url: Callable[[str], str],
    evidence_by_claim: dict | None = None,
    # Optional overrides
    slots_strategy: Callable[[str], set[str]] | None = None,
    required_slots_strategy: Callable[[str], set[str]] | None = None,
    merge_covers_strategy: Callable[[list[str] | None, set[str]], set[str]] | None = None,
    claim_sig_strategy: Callable[[dict[str, Any]], Any] | None = None,
    ev_sig_strategy: Callable[[dict[str, Any]], Any] | None = None,
    sig_compat_strategy: Callable[[Any, Any], bool] | None = None,
):
    return compute_spillover(
        sources=sources,
        claims=claims,
        cluster_map=cluster_map,
        top_k=top_k,
        normalize_url=normalize_url,
        slots_from_assertion_key=slots_strategy or slots_from_assertion_key,
        required_slots_for_verification_target=required_slots_strategy or required_slots_for_verification_target,
        merge_covers=merge_covers_strategy or merge_covers,
        claim_event_signature=claim_sig_strategy or claim_event_signature,
        evidence_event_signature=ev_sig_strategy or evidence_event_signature,
        signature_compatible=sig_compat_strategy or signature_compatible,
        evidence_by_claim=evidence_by_claim,
    )
