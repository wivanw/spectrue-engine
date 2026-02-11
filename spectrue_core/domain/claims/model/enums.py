"""
Core enums for claim domain model.
"""
from enum import Enum

class ClaimRole(str, Enum):
    """
    Role of a claim in the document structure.
    
    Affects RGBA aggregation weight and explanation inclusion.
    - CORE/SUPPORT/ATTRIBUTION: Full weight in scoring
    - CONTEXT/META: Weight=0, explain-only
    """
    CORE = "core"
    """Central claim of the article. Highest verification priority."""

    SUPPORT = "support"
    """Supporting evidence for a core claim."""

    CONTEXT = "context"
    """Background information. Not a fact to verify."""

    META = "meta"
    """Information about the article/source itself."""

    ATTRIBUTION = "attribution"
    """Quote attribution: who said what."""

    AGGREGATED = "aggregated"
    """Summary derived from multiple sources."""

    SUBCLAIM = "subclaim"
    """Subordinate detail within a larger claim."""

    # Structured document roles
    THESIS = "thesis"
    """Main thesis or conclusion of the article."""

    BACKGROUND = "background"
    """Background context for the topic."""

    EXAMPLE = "example"
    """Illustrative example supporting another claim."""

    HEDGE = "hedge"
    """Hedged/qualified statement ("may", "might", "possibly")."""

    COUNTERCLAIM = "counterclaim"
    """Counterpoint or opposing claim."""

    DEFINITION = "definition"
    """Term definition or concept explanation."""

    FORECAST = "forecast"
    """Prediction or forecast about future events. Limited verifiability."""


class VerificationTarget(str, Enum):
    """
    What aspect of the claim to verify.
    
    Determines search strategy and verdict semantics.
    - REALITY: Is it factually true? (standard verification)
    - ATTRIBUTION: Did X really say Y?
    - EXISTENCE: Does the source/document exist?
    - NONE: Not verifiable (predictions, opinions)
    """
    REALITY = "reality"
    """Verify factual accuracy against reality. Standard path."""

    ATTRIBUTION = "attribution"
    """Verify that person X said/did thing Y."""

    EXISTENCE = "existence"
    """Verify that a source/document/entity exists."""

    NONE = "none"
    """Not verifiable. Predictions, opinions, horoscopes, subjective."""


class MetadataConfidence(str, Enum):
    """
    Confidence in the extracted metadata.
    
    Used for fail-open decisions:
    - LOW: Trigger fail-open (don't skip, do Phase A-light)
    - MEDIUM: Normal processing
    - HIGH: Trust metadata fully
    """
    LOW = "low"
    """Low confidence. Trigger fail-open: always do Phase A-light."""

    MEDIUM = "medium"
    """Medium confidence. Normal processing."""

    HIGH = "high"
    """High confidence. Trust metadata fully."""


class Dimension(str, Enum):
    """
    Assertion dimension - determines verification behavior.
    """
    FACT = "FACT"
    """Must be proven/refuted by evidence. Strict verification."""

    CONTEXT = "CONTEXT"
    """Contextual framing (time zone, audience). Informational only."""

    INTERPRETATION = "INTERPRETATION"
    """Parser interpretation from ambiguous text. Bounded, flagged."""


class VerificationScope(str, Enum):
    """How strictly to verify this assertion."""
    STRICT = "STRICT"
    """For FACT assertions - can be VERIFIED/REFUTED/AMBIGUOUS."""

    SOFT = "SOFT"
    """For CONTEXT - only VERIFIED/AMBIGUOUS unless explicitly contradicted."""


class ClaimDomain(str, Enum):
    """High-level domain of the claim."""
    NEWS = "news"
    SCIENCE = "science"
    POLITICS = "politics"
    FINANCE = "finance"
    HEALTH = "health"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    ENTERTAINMENT = "entertainment"
    HISTORY = "history"
    OTHER = "other"


class ClaimType(str, Enum):
    """Type of claim for search/verification strategy."""
    EVENT = "event"
    """Something happened at a time/place."""

    ATTRIBUTION = "attribution"
    """Someone said/did something."""

    NUMERIC = "numeric"
    """Specific numbers, statistics, measurements."""

    DEFINITION = "definition"
    """What something is/means."""

    COMPARISON = "comparison"
    """X is greater/less/equal to Y."""

    POLICY = "policy"
    """Rules, laws, regulations."""

    TIMELINE = "timeline"
    """Sequence of events, dates."""

    BIOGRAPHY = "biography"
    """Facts about a person."""

    OTHER = "other"


class ClaimStructureType(str, Enum):
    """Logical structure type of the claim."""
    EMPIRICAL_NUMERIC = "empirical_numeric"
    EVENT = "event"
    CAUSAL = "causal"
    ATTRIBUTION = "attribution"
    DEFINITION = "definition"
    POLICY_PLAN = "policy_plan"
    FORECAST = "forecast"
    EXISTENCE = "existence"
    META_SCIENTIFIC = "meta_scientific"
    OTHER = "other"
