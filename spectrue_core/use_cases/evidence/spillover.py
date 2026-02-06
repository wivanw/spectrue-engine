"""Evidence spillover use cases."""

from __future__ import annotations

from typing import Any, Callable

from spectrue_core.domain.evidence.spillover import compute_spillover


def run_spillover(*, sources: list[dict], claims: list[dict], cluster_map: dict[str, str], top_k: int, normalize_url: Callable[[str], str], slots_from_assertion_key: Callable[[str], set[str]], required_slots_for_verification_target: Callable[[str], set[str]], merge_covers: Callable[[list[str] | None, set[str]], set[str]], claim_event_signature: Callable[[dict[str, Any]], Any], evidence_event_signature: Callable[[dict[str, Any]], Any], signature_compatible: Callable[[Any, Any], bool], evidence_by_claim: dict | None = None):
    return compute_spillover(
        sources=sources,
        claims=claims,
        cluster_map=cluster_map,
        top_k=top_k,
        normalize_url=normalize_url,
        slots_from_assertion_key=slots_from_assertion_key,
        required_slots_for_verification_target=required_slots_for_verification_target,
        merge_covers=merge_covers,
        claim_event_signature=claim_event_signature,
        evidence_event_signature=evidence_event_signature,
        signature_compatible=signature_compatible,
        evidence_by_claim=evidence_by_claim,
    )
