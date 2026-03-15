from __future__ import annotations

from enum import Enum


class AnalysisMode(str, Enum):
    """API-facing analysis mode names."""
    GENERAL = "general"
    DEEP = "deep"
    DEEP_V2 = "deep_v2"


class ScoringMode(str, Enum):
    """Scoring validation modes."""
    STANDARD = "standard"
    DEEP = "deep"


class SearchProfileName(str, Enum):
    """Search policy profile names."""
    GENERAL = "general"
    STANDARD = "general"  # Alias
    DEEP = "deep"


class SearchDepth(str, Enum):
    """Search depth levels for retrieval."""
    BASIC = "basic"
    ADVANCED = "advanced"


class StancePassMode(str, Enum):
    """Stance detection pass modes."""
    SINGLE = "single"
    TWO_PASS = "two_pass"


class RelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ENTAILS = "entails"


class VerdictStatus(str, Enum):
    """Verdict outcome for an assertion or claim."""
    VERIFIED = "verified"
    REFUTED = "refuted"
    AMBIGUOUS = "ambiguous"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    SATIRICAL = "satirical"

    @classmethod
    def from_score(cls, score: float) -> VerdictStatus:
        """Derive verdict label from LLM score."""
        if score > 0.65:
            return cls.VERIFIED
        elif score < 0.35:
            return cls.REFUTED
        else:
            return cls.AMBIGUOUS


class VerdictState(str, Enum):
    """Tier-dominant verdict state independent from score."""
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONFLICTED = "conflicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"

    @classmethod
    def from_scores(cls, llm_score: float, n_support: int, n_refute: int) -> VerdictState:
        """Derive canonical verdict state from LLM score and evidence counts."""
        if llm_score > 0.65:
            return cls.SUPPORTED
        elif llm_score < 0.35:
            return cls.REFUTED
        elif n_support > 0 or n_refute > 0:
            return cls.CONFLICTED
        else:
            return cls.INSUFFICIENT_EVIDENCE
