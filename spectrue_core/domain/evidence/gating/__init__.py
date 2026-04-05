"""Gating logic package."""

from .models import GateDecisionPayload
from .uncertainty import compute_beta_uncertainty, extract_evidence_features
from .cost import estimate_stance_cost, estimate_cluster_cost
from .logic import compute_stance_gate, compute_cluster_gate
from spectrue_core.domain.evidence.weights import (
    StanceWeights,
    ClusterWeights,
    DEFAULT_STANCE_WEIGHTS,
    DEFAULT_CLUSTER_WEIGHTS,
)

__all__ = [
    "GateDecisionPayload",
    "compute_beta_uncertainty",
    "extract_evidence_features",
    "estimate_stance_cost",
    "estimate_cluster_cost",
    "compute_stance_gate",
    "compute_cluster_gate",
    "StanceWeights",
    "ClusterWeights",
    "DEFAULT_STANCE_WEIGHTS",
    "DEFAULT_CLUSTER_WEIGHTS",
]
