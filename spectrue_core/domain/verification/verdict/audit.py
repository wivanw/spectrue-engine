from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ClaimAudit:
    claim_id: str
    predicate_type: Literal[
        "event",
        "measurement",
        "quote",
        "policy",
        "ranking",
        "causal",
        "other",
    ]
    truth_conditions: list[str]
    expected_evidence_types: list[str]
    failure_modes: list[str]
    assertion_strength: Literal["weak", "medium", "strong"]
    risk_facets: list[str]
    honesty_facets: list[str]
    what_would_change_mind: list[str]
    audit_confidence: float


@dataclass
class EvidenceAudit:
    claim_id: str
    evidence_id: str
    source_id: str
    stance: Literal["support", "refute", "unclear", "unrelated"]
    directness: Literal["direct", "indirect", "tangential"]
    specificity: Literal["high", "medium", "low"]
    quote_integrity: Literal["ok", "partial", "out_of_context", "not_applicable"]
    extraction_confidence: float
    novelty_vs_copy: Literal["original", "syndicated", "unknown"]
    dependency_hints: list[str]
    audit_confidence: float
