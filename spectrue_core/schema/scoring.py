from __future__ import annotations

from typing import Optional
from pydantic import Field

from spectrue_core.schema.serialization import SchemaModel
from spectrue_core.domain.verification.verdict.model import (
    AnalysisMode,
    ScoringMode,
    VerdictStatus,
    VerdictState,
    BeliefState as DomainBeliefState,
    ConsensusState as DomainConsensusState,
    ClaimNode as DomainClaimNode,
    ClaimEdge as DomainClaimEdge,
    ScoringTraceStep as DomainScoringTraceStep,
    log_odds_to_prob,
)
from spectrue_core.domain.claims.model import ClaimRole
from spectrue_core.domain.claims.graph.types import EdgeRelation as RelationType

__all__ = [
    "AnalysisMode",
    "ScoringMode",
    "VerdictStatus",
    "VerdictState",
    "RelationType",
    "BeliefState",
    "ClaimNode",
    "ClaimEdge",
    "ClaimNode",
    "ClaimEdge",
    "ScoringTraceStep",
    "ConsensusState",
    "ClaimRole",
]


class BeliefState(SchemaModel, DomainBeliefState):
    log_odds: float = Field(..., description="Belief in log-odds space")
    confidence: float = Field(0.0, description="Measure of certainty/variance")

    @property
    def probability(self) -> float:
        return log_odds_to_prob(self.log_odds)


class ClaimNode(SchemaModel, DomainClaimNode):
    claim_id: str
    text: str
    role: str
    local_belief: Optional[BeliefState] = None
    propagated_belief: Optional[BeliefState] = None


class ClaimEdge(SchemaModel, DomainClaimEdge):
    source_id: str
    target_id: str
    relation: RelationType | str
    weight: float = Field(..., ge=0.0, le=1.0, description="Semantic strength of the connection")


class ScoringTraceStep(SchemaModel, DomainScoringTraceStep):
    step_id: int
    description: str
    delta: float = Field(..., description="Change in log-odds")
    new_belief: float = Field(..., description="Resulting log-odds")


class ConsensusState(SchemaModel, DomainConsensusState):
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized consensus level")
    stability: float = Field(..., description="Temporal stability")
    source_count: int = Field(..., description="Number of independent sources")