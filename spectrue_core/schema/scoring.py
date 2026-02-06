from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import Field

from spectrue_core.schema.serialization import SchemaModel
from spectrue_core.domain.verification.verdict.model import (
    BeliefState as DomainBeliefState,
    ConsensusState as DomainConsensusState,
    log_odds_to_prob,
)

__all__ = [
    "ClaimRole",
    "RelationType",
    "BeliefState",
    "ClaimNode",
    "ClaimEdge",
    "ScoringTraceStep",
    "ConsensusState",
]

class ClaimRole(str, Enum):
    THESIS = "thesis"
    SUPPORT = "support"
    BACKGROUND = "background"
    COUNTER = "counter"

class RelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ENTAILS = "entails"

class BeliefState(SchemaModel, DomainBeliefState):
    log_odds: float = Field(..., description="Belief in log-odds space")
    confidence: float = Field(0.0, description="Measure of certainty/variance")

    @property
    def probability(self) -> float:
        return log_odds_to_prob(self.log_odds)

class ClaimNode(SchemaModel):
    claim_id: str
    text: str
    role: ClaimRole
    local_belief: Optional[BeliefState] = None
    propagated_belief: Optional[BeliefState] = None

class ClaimEdge(SchemaModel):
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = Field(..., ge=0.0, le=1.0, description="Semantic strength of the connection")

class ScoringTraceStep(SchemaModel):
    step_id: int
    description: str
    delta: float = Field(..., description="Change in log-odds")
    new_belief: float = Field(..., description="Resulting log-odds")

class ConsensusState(SchemaModel, DomainConsensusState):
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized consensus level")
    stability: float = Field(..., description="Temporal stability")
    source_count: int = Field(..., description="Number of independent sources")
