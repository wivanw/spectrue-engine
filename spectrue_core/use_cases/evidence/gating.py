"""Evidence gating use cases."""

from __future__ import annotations

from spectrue_core.domain.evidence.gating import (
    compute_beta_uncertainty,
    compute_cluster_gate,
    compute_stance_gate,
    extract_evidence_features,
    ClusterWeights,
    StanceWeights,
)

__all__ = [
    "compute_gates",
    "ClusterWeights",
    "StanceWeights",
]


def compute_gates(*, evidence_index, claims: list[dict], ledger, stance_weights: StanceWeights, cluster_weights: ClusterWeights):
    features = extract_evidence_features(evidence_index, claims)
    uncertainty = compute_beta_uncertainty(evidence_index)
    stance_gate_payload = compute_stance_gate(features, uncertainty, ledger, stance_weights)
    cluster_gate_payload = compute_cluster_gate(
        features,
        stance_gate_payload.enabled,
        ledger,
        cluster_weights,
    )
    return features, uncertainty, stance_gate_payload, cluster_gate_payload
