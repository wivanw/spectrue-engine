from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ─────────────────────────────────────────────────────────────────────────────
# Sufficiency Types (Domain)
# ─────────────────────────────────────────────────────────────────────────────

class SufficiencyStatus(str, Enum):
    """Result of sufficiency check."""
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    SKIP = "skip"


class SufficiencyDecision(str, Enum):
    """High-level decision for iterative retrieval."""
    ENOUGH = "ENOUGH"
    NEED_FOLLOWUP = "NEED_FOLLOWUP"
    STOP = "STOP"


@dataclass
class SufficiencyResult:
    """Result of evidence sufficiency check for a claim."""
    claim_id: str
    status: SufficiencyStatus = SufficiencyStatus.INSUFFICIENT
    reason: str = ""
    rule_matched: str = ""
    authoritative_count: int = 0
    reputable_count: int = 0
    independent_domains: int = 0
    has_quotes: bool = False
    support_refute_count: int = 0
    context_only_count: int = 0
    posterior_p: float = 0.5


@dataclass
class SufficiencyDecisionResult:
    """Decision returned by the sufficiency judge."""
    claim_id: str
    decision: SufficiencyDecision
    reason: str
    rule_matched: str = ""
    degraded_confidence: bool = False
    coverage: float = 0.0
    diversity: float = 0.0
