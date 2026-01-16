# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import math

class ClaimRole(str, Enum):
    THESIS = "thesis"
    SUPPORT = "support"
    BACKGROUND = "background"
    COUNTER = "counter"

class RelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ENTAILS = "entails"

class BeliefState(BaseModel):
    log_odds: float = Field(..., description="Belief in log-odds space")
    confidence: float = Field(0.0, description="Measure of certainty/variance")

    @property
    def probability(self) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-self.log_odds))
        except OverflowError:
            return 0.0 if self.log_odds < 0 else 1.0

class ClaimNode(BaseModel):
    claim_id: str
    text: str
    role: ClaimRole
    local_belief: Optional[BeliefState] = None
    propagated_belief: Optional[BeliefState] = None

class ClaimEdge(BaseModel):
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = Field(..., ge=0.0, le=1.0, description="Semantic strength of the connection")

class ScoringTraceStep(BaseModel):
    step_id: int
    description: str
    delta: float = Field(..., description="Change in log-odds")
    new_belief: float = Field(..., description="Resulting log-odds")

class ConsensusState(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized consensus level")
    stability: float = Field(..., description="Temporal stability")
    source_count: int = Field(..., description="Number of independent sources")
