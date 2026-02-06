# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Stance domain models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StanceFeatures:
    """Structural features for stance posterior calculation."""

    # LLM observations (noisy)
    llm_stance: str  # Raw LLM prediction
    llm_relevance: Optional[float]  # 0-1 or None if missing

    # Structural signals (deterministic)
    quote_present: bool
    has_evidence_chunk: bool
    source_prior: float  # 0-1, derived from tier/domain quality

    # Optional: retrieval signals
    retrieval_rank: int = 0  # Position in search results (0 = first)
    retrieval_score: Optional[float] = None  # Search API score if available


@dataclass
class StancePosterior:
    """Posterior probability distribution over stance classes."""

    p_support: float
    p_refute: float
    p_neutral: float
    p_context: float
    p_irrelevant: float

    # Derived metrics
    p_evidence: float  # P(S ∈ {SUPPORT, REFUTE})
    argmax_stance: str  # Most likely stance
    entropy: float  # Uncertainty measure

    def to_dict(self) -> dict:
        return {
            "p_support": round(self.p_support, 4),
            "p_refute": round(self.p_refute, 4),
            "p_neutral": round(self.p_neutral, 4),
            "p_context": round(self.p_context, 4),
            "p_irrelevant": round(self.p_irrelevant, 4),
            "p_evidence": round(self.p_evidence, 4),
            "argmax_stance": self.argmax_stance,
            "entropy": round(self.entropy, 4),
        }
