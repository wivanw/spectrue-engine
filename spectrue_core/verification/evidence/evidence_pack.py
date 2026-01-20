# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Evidence Pack data structures for LLM-centric scoring.

The Evidence Pack is the structured input to the LLM scorer.
Code collects evidence and computes quality metrics; LLM produces verdict.

Philosophy:
- Code = controller of evidence quality (metrics, caps)
- LLM = judge with constraints (verdict, explanation)
"""

from typing import Literal, TypedDict, Any

from spectrue_core.utils.trace import Trace
from spectrue_core.verification.calibration.calibration_models import logistic_score
from spectrue_core.verification.calibration.calibration_registry import CalibrationRegistry

def _has_evidence_chunk(source: Any) -> bool:
    """Check whether a source includes a usable evidence chunk."""
    if not isinstance(source, dict):
        return False
    return bool(
        source.get("quote")
        or source.get("snippet")
        or source.get("content")
        or source.get("content_excerpt")
        or source.get("key_snippet")
    )



# ─────────────────────────────────────────────────────────────────────────────
# Claim Types
# ─────────────────────────────────────────────────────────────────────────────

ClaimType = Literal[
    "core",
    "numeric",
    "timeline",
    "attribution",
    "sidefact",
    "atomic",
    "causal",
    "comparative",
    "policy_plan",
    "definition",
    "future",
    "existence",
]
"""
Claim types for multi-claim extraction:
- core: Main factual assertion
- numeric: Claims with specific numbers/statistics
- timeline: Claims about dates, deadlines, sequences
- attribution: Claims about who said/did something
- sidefact: Secondary supporting facts
"""

ClaimRoleType = Literal[
    "core", "support", "context", "meta", "attribution", "aggregated", "subclaim",
    "thesis", "background", "example", "hedge", "counterclaim",
    "peripheral",
]
"""Role of a claim in document structure."""

ClaimStructureType = Literal[
    "empirical_numeric", "event", "causal", "attribution",
    "definition", "policy_plan", "forecast", "meta_scientific",
    "comparative", "future", "existence",
]
"""Logical structure type of a claim."""

Stance = Literal["support", "contradict", "neutral", "unclear", "context", "irrelevant", "mention"]
"""Position of a source relative to a claim. Upper/Lowercase handled by runtime normalization."""

VerificationTarget = Literal["reality", "attribution", "existence", "none"]

EvidenceChannel = Literal["authoritative", "reputable_news", "local_media", "social", "low_reliability"]

EvidenceStance = Literal["SUPPORT", "REFUTE", "CONTEXT", "IRRELEVANT"]

ConfidenceLevel = Literal["low", "medium", "high"]

VerdictLabel = Literal["verified", "refuted", "ambiguous", "insufficient"]

SourceType = Literal[
    "primary",        # Original source (official website, press release)
    "official",       # Official statement (gov, company, org)
    "independent_media",  # Independent news outlet
    "aggregator",     # News aggregator, syndication
    "social",         # Social media post
    "fact_check",     # Fact-check from Oracle (Google Fact Check API)
    "unknown",        # Cannot determine
]
"""Classification of source authority."""


# ─────────────────────────────────────────────────────────────────────────────
# Oracle Check Result
# ─────────────────────────────────────────────────────────────────────────────

OracleStatus = Literal["CONFIRMED", "REFUTED", "MIXED", "EMPTY", "ERROR", "DISABLED"]
"""
Oracle verdict status:
- CONFIRMED: Fact-check confirms the claim is true
- REFUTED: Fact-check says the claim is false/fake
- MIXED: Fact-check says partially true or needs context
- EMPTY: No relevant fact-check found
- ERROR: API failure (check error_status_code)
- DISABLED: Oracle validator not configured
"""

ArticleIntent = Literal["news", "evergreen", "official", "opinion", "prediction"]
"""
Article intent classification for Oracle triggering:
- news: Current events, breaking news (CHECK Oracle)
- evergreen: Science facts, historical claims, health info (CHECK Oracle)
- official: Government/company announcements (CHECK Oracle)
- opinion: Editorial, commentary (SKIP Oracle)
- prediction: Future events (SKIP Oracle)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Query Candidates for Round-Robin Selection
# ─────────────────────────────────────────────────────────────────────────────

QueryRole = Literal["CORE", "NUMERIC", "ATTRIBUTION", "LOCAL"]
"""
Query role types for Coverage Engine:
- CORE: Event + Date + Action (Required, highest priority)
- NUMERIC: Metric + Value + "Official Data" (If numbers exist)
- ATTRIBUTION: Person + "Quote" + Source (If quotes exist)
- LOCAL: Local language variant of CORE query
"""

# Priority scores for query roles (used in round-robin selection)
QUERY_ROLE_SCORES: dict[str, float] = {
    "CORE": 1.0,        # Most important - verifies event existence
    "NUMERIC": 0.8,     # Verifies specific numbers/statistics
    "ATTRIBUTION": 0.7, # Verifies quotes/sources
    "LOCAL": 0.5,       # Alternative language phrasing
}


class QueryCandidate(TypedDict, total=False):
    """
    A typed search query candidate with priority score.
    
    Used by the Coverage Engine for topic-aware round-robin selection.
    The 'role' determines priority: CORE queries are selected first (Pass 1),
    then NUMERIC/ATTRIBUTION (Pass 2), then LOCAL (Pass 3).
    """
    text: str           # The search query text
    role: QueryRole     # Query type/role
    score: float        # Priority score (default from QUERY_ROLE_SCORES)
    lang: str           # Target language (optional, e.g., "en", "uk")


class OracleCheckResult(TypedDict, total=False):
    """
    Result from Google Fact Check API with LLM semantic validation.
    
    Used in hybrid Oracle flow:
    - JACKPOT (relevance > 0.9): Stop pipeline, return immediately
    - EVIDENCE (0.5 < relevance <= 0.9): Add to evidence pack, continue search
    - MISS (relevance <= 0.5 or EMPTY): Ignore, proceed to standard search
    """
    status: OracleStatus              # Verdict from fact-check
    url: str | None                   # URL of the fact-check article
    claim_reviewed: str | None        # The claim text from the external fact-check
    summary: str | None               # The verdict/explanation
    relevance_score: float            # 0.0 to 1.0 (Calculated by LLM)
    is_jackpot: bool                  # True if relevance > 0.9 (Stop search immediately)
    publisher: str | None             # Fact-check publisher name (Snopes, PolitiFact, etc.)
    rating: str | None                # Original textual rating from fact-checker
    source_provider: str | None       # UX: "Snopes via Google Fact Check"
    # M73.4: Error fields (when status == ERROR)
    error_status_code: int | None     # HTTP status code on failure
    error_detail: str | None          # Error message


# ─────────────────────────────────────────────────────────────────────────────
# Article Context
# ─────────────────────────────────────────────────────────────────────────────

class ArticleContext(TypedDict, total=False):
    """Context about the article being verified."""
    url: str | None
    title: str | None
    publisher: str | None
    published_at: str | None
    content_lang: str | None
    text_excerpt: str  # First ~500 chars for LLM context


# ─────────────────────────────────────────────────────────────────────────────
# Claim Structure
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAnchor(TypedDict, total=False):
    """Normalized time window for a claim."""
    window: dict[str, str | None]
    granularity: Literal["day", "month", "year", "relative", "unknown"]
    claim_time_label: Literal["past", "present", "future", "atemporal"] | None
    is_time_sensitive: bool


class LocalePlan(TypedDict, total=False):
    """Locale preferences for UI, content, and context."""
    ui_locale: str | None
    content_lang: str | None
    context_lang: str | None
    primary: str
    fallbacks: list[str]


def _evidence_feature_row(src: dict) -> dict[str, float]:
    quote = src.get("quote") or ""
    snippet = src.get("snippet") or src.get("content") or ""
    text = quote or snippet or ""
    has_digits = 1.0 if isinstance(text, str) and any(ch.isdigit() for ch in text) else 0.0
    return {
        "has_quote": 1.0 if quote else 0.0,
        "has_chunk": 1.0 if _has_evidence_chunk(src) else 0.0,
        "has_digits": has_digits,
        "quote_len_norm": min(1.0, len(str(quote)) / 200.0) if quote else 0.0,
        "snippet_len_norm": min(1.0, len(str(snippet)) / 400.0) if snippet else 0.0,
        "extraction_success": 1.0 if snippet or quote else 0.0,
    }


def score_evidence_likeness(
    sources: list[dict],
    *,
    calibration_registry: CalibrationRegistry | None = None,
) -> float:
    """
    Calibrated evidence-likeness scoring for retrieval evaluation.
    """
    if not sources:
        return 0.0

    registry = calibration_registry or CalibrationRegistry.from_runtime(None)
    model = registry.get_model("evidence_likeness")
    scores: list[float] = []
    samples: list[dict] = []

    for src in sources:
        if not isinstance(src, dict):
            continue
        features = _evidence_feature_row(src)
        if model:
            score, trace = model.score(features)
        else:
            policy = registry.policy.evidence_likeness
            raw, score = logistic_score(
                features,
                policy.fallback_weights or policy.weights,
                bias=policy.fallback_bias or policy.bias,
            )
            trace = {
                "model": "evidence_likeness",
                "version": policy.version,
                "features": features,
                "weights": policy.fallback_weights or policy.weights,
                "bias": policy.fallback_bias or policy.bias,
                "raw_score": raw,
                "score": score,
                "fallback_used": True,
            }
        scores.append(float(score))
        if len(samples) < 5:
            samples.append(
                {
                    "domain": src.get("domain"),
                    "score": float(score),
                    "trace": trace,
                }
            )

    avg = sum(scores) / len(scores) if scores else 0.0
    Trace.event(
        "evidence_likeness.scored",
        {
            "avg_score": float(avg),
            "sample_count": len(samples),
            "samples": samples,
        },
    )
    return float(avg)


class EvidenceNeed(TypedDict, total=False):
    """Claim-level signal for evidence depth needs."""
    level: str | None

class ClaimAnchor(TypedDict, total=False):
    """Tracks claim's position in original text."""
    chunk_id: str
    char_start: int
    char_end: int
    section_path: list[str]


class EvidenceRequirement(TypedDict, total=False):
    """What evidence is required to verify this claim."""
    needs_primary_source: bool      # Needs official/primary confirmation
    needs_independent_2x: bool      # Needs 2+ independent sources
    needs_exact_quote: bool         # Needs verbatim quote match
    needs_recent_source: bool       # Needs source from last N days
    max_age_days: int | None        # Max acceptable source age


class ClaimStructure(TypedDict, total=False):
    """Structured representation of claim logic."""
    type: ClaimStructureType
    premises: list[str]
    conclusion: str | None
    dependencies: list[str]


class Claim(TypedDict, total=False):
    """A single atomic claim extracted from the article."""
    id: str                         # Unique claim ID (c1, c2, ...)
    text: str                       # The claim text (may contain pronouns)
    language: str                   # ISO-639-1 language code (e.g., "en", "uk")
    type: ClaimType                 # Claim classification
    importance: float               # 0-1, how critical to main thesis
    evidence_requirement: EvidenceRequirement
    search_queries: list[str]       # Generated queries for this claim (legacy)
    check_oracle: bool              # T10: Should this specific claim be checked against Oracle?
    # Retrieval escalation fields (from LLM extraction)
    subject_entities: list[str]     # Named entities (people, orgs, places)
    retrieval_seed_terms: list[str] # Keywords for search queries
    context_entities: list[str]     # Document-level context terms
    time_anchor: dict               # Temporal anchor (type, value, start, end)
    # Context-aware atomization fields
    normalized_text: str            # Self-sufficient statement with pronouns resolved
    topic_group: str                # Topic tag (e.g., "Economy", "War", "Science")
    check_worthiness: float         # 0-1, how important to verify (filters opinions)
    # Topic-Aware Round-Robin fields
    topic_key: str                  # Specific entity tag (e.g., "Fomalhaut System", "Bitcoin Price")
    query_candidates: list["QueryCandidate"]  # Typed query candidates with roles
    # Smart Routing method
    search_method: Literal["news", "general_search", "academic"]
    # Layer 4: Evidence-Need Routing
    evidence_need: Literal[
        "empirical_study", "guideline", "official_stats",
        "expert_opinion", "anecdotal", "news_report", "unknown"
    ]
    # Safety & Coverage
    anchor: ClaimAnchor
    text_safe: str
    is_actionable_medical: bool
    danger_tags: list[str]
    redacted_spans: list[dict]
    # Salience
    harm_potential: int  # 1-5 scale (5=Highest Harm Risk)
    # Claim Category (Satire Detection)
    claim_category: Literal["FACTUAL", "SATIRE", "OPINION", "HYPERBOLIC"]
    satire_likelihood: float  # 0.0-1.0, probability claim is satirical
    # Claim structure + role
    claim_role: ClaimRoleType
    structure: ClaimStructure
    # Claim-Centric Orchestration Metadata
    # Optional: When present, enables metadata-driven routing
    # Import: from spectrue_core.schema.claim_metadata import ClaimMetadata
    metadata: Any  # ClaimMetadata | None (use Any to avoid circular import)
    # Core data contract alignment
    verification_target: VerificationTarget
    role: ClaimRoleType
    temporality: TemporalAnchor
    locale_plan: LocalePlan
    metadata_confidence: ConfidenceLevel
    priority_score: float
    centrality: float
    tension: float




# ─────────────────────────────────────────────────────────────────────────────
# Search Result Structure
# ─────────────────────────────────────────────────────────────────────────────

class SearchResult(TypedDict, total=False):
    """A single search result with stance and quality annotations."""
    claim_id: str | None          # Which claim this result pertains to
    url: str
    domain: str                     # Registrable domain (example.com)
    title: str
    snippet: str                    # Search result snippet
    content_excerpt: str | None     # Extracted content (if fetched)
    published_at: str | None
    source_type: SourceType
    stance: Stance                  # Position relative to claim
    quote: str | None               # Verified quote for the stance
    relevance_score: float          # 0-1, how relevant to claim
    timeliness_status: str | None   # in_window | outdated | unknown_date
    key_snippet: str | None         # Most relevant quote from content
    quote_matches: list[str]        # Exact quotes that match claim
    is_trusted: bool                # From trusted sources registry
    is_duplicate: bool              # Content duplicate of another result
    duplicate_of: str | None        # URL of original if duplicate
    evidence_tier: str | None       # A, A', B, C, D (if known)
    pass_type: str | None           # SUPPORT_ONLY, REFUTE_ONLY, SINGLE_PASS
    quote_span: str | None          # Quote aligned to SUPPORT
    contradiction_span: str | None  # Quote aligned to REFUTE
    evidence_refs: list[str]        # Source URLs or references
    stance_confidence: str | None   # "low" for low-tier SUPPORT

    # Evidence metadata (typed, non-heuristic)
    evidence_role: Literal["direct", "indirect", "mention_only"] | None
    covers: list[str] | None

    # Assertion-level mapping
    assertion_key: str | None       # Which assertion this evidence maps to (e.g., "event.location.city")

    # Content availability status
    content_status: Literal["available", "unavailable", "blocked", "error"]
    unavailable_reason: str | None  # Why content couldn't be retrieved

    # Bayesian stance posterior (M113+)
    # Soft probabilities instead of hard stance labels
    p_support: float                # P(S* = SUPPORT | features)
    p_refute: float                 # P(S* = REFUTE | features)
    p_neutral: float                # P(S* = NEUTRAL | features)
    p_evidence: float               # P(S* ∈ {SUPPORT, REFUTE})
    posterior_entropy: float        # Uncertainty in stance classification


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Item (Canonical Scoring Contract)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceItem(TypedDict, total=False):
    """Canonical evidence item for deterministic scoring."""
    url: str
    domain: str
    title: str | None
    snippet: str | None
    channel: EvidenceChannel
    tier: Literal["A", "B", "C", "D"]
    tier_reason: str | None
    claim_id: str | None
    stance: EvidenceStance
    quote: str | None
    relevance: float
    published_at: str | None
    temporal_flag: Literal["in_window", "outdated", "unknown"]
    fetched: bool | None
    raw_text_chars: int | None
    r_domain: float | None
    r_contextual: float | None
    r_eff: float | None
    has_authority_anchor: bool
    authority_anchor_reason: str | None
    reliability_confidence: Literal["low", "medium", "high"] | None


class EvidencePackStats(TypedDict, total=False):
    """Aggregate statistics for an evidence pack."""
    domain_diversity: int
    tiers_present: dict[str, int]
    support_count: int
    refute_count: int
    context_count: int
    outdated_ratio: float


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

class ClaimMetrics(TypedDict, total=False):
    """Metrics for a single claim's evidence quality."""
    independent_domains: int        # Number of unique domains
    primary_present: bool           # Is there a primary source?
    official_present: bool          # Is there an official source?
    stance_distribution: dict[str, int]  # {"support": 3, "contradict": 1, ...}
    coverage: float                 # 0-1, how well is claim covered?
    freshness_days_median: int | None
    # Additional claim metadata for scoring
    topic_group: str | None         # Topic category from claim extraction
    claim_type: str | None          # Claim type (core, numeric, etc.)


class AssertionMetrics(TypedDict, total=False):
    """Metrics for a single assertion (field-level fact)."""
    support_count: int              # Number of supporting sources
    refute_count: int               # Number of refuting sources
    tier_coverage: dict[str, int]   # Tiers present (A, B, C, D)
    primary_present: bool
    official_present: bool
    content_unavailable_count: int  # Count of sources with unavailable content


class EvidenceGap(TypedDict, total=False):
    """A detected gap in evidence for a claim."""
    claim_id: str                   # Which claim has the gap
    gap_type: str                   # missing_primary, insufficient_sources, no_contradiction, etc.
    description: str                # Human-readable description
    severity: float                 # 0-1, how critical is this gap
    suggested_queries: list[str]    # Queries to fill the gap


class EvidenceMetrics(TypedDict, total=False):
    """Aggregate metrics across all claims."""
    total_sources: int
    unique_domains: int
    duplicate_ratio: float          # % of sources that are duplicates
    per_claim: dict[str, ClaimMetrics]  # claim_id -> metrics
    per_assertion: dict[str, AssertionMetrics]  # assertion_key -> metrics (M70)
    overall_coverage: float         # Average coverage across claims
    freshness_days_median: int | None
    source_type_distribution: dict[str, int]


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Constraints (Code-Enforced Caps)
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceConstraints(TypedDict, total=False):
    """
    Code-enforced caps on LLM's confidence.

    These caps are based on evidence quality, not tier math.
    LLM must respect these caps; code enforces post-LLM.
    """
    # Per-claim caps
    cap_per_claim: dict[str, float]  # claim_id -> max confidence

    # Global cap (minimum of all claim caps)
    global_cap: float

    # Reasons for capping (for explainability)
    cap_reasons: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Complete Evidence Pack
# ─────────────────────────────────────────────────────────────────────────────

class EvidencePack(TypedDict, total=False):
    """
    Complete evidence package for LLM scoring.

    This is the contract between code (evidence collector) and LLM (judge).
    Code computes all metrics and caps; LLM produces verdict within caps.
    """
    # Context
    article: ArticleContext
    original_fact: str              # Original text submitted for verification

    # Claims (multi-claim extraction)
    claims: list[Claim]
    claim_units: list[Any] | None   # Structured ClaimUnits (Pydantic objects)

    # Evidence
    search_results: list[SearchResult]  # All sources (scored + context)
    scored_sources: list[SearchResult]  # Sources contributing to verdict
    context_sources: list[SearchResult] # Sources retained for context/transparency

    # Metrics (code-computed)
    metrics: EvidenceMetrics

    # Constraints (code-enforced)
    constraints: ConfidenceConstraints

    # Canonical scoring inputs
    claim_id: str
    items: list[EvidenceItem]
    stats: EvidencePackStats
    global_cap: float
    cap_reasons: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# LLM Scoring Output
# ─────────────────────────────────────────────────────────────────────────────

class ClaimVerdict(TypedDict, total=False):
    """Verdict for a single claim from LLM."""
    claim_id: str
    verdict: VerdictLabel | Literal["unverified", "partially_verified"]
    evidence_strength: float        # 0-1, how well supported
    source_independence: float      # 0-1, are sources independent?
    attribution_integrity: float    # 0-1, is attribution correct?
    confidence: float | ConfidenceLevel  # 0-1 legacy or label
    rationale: str                  # Explanation with source citations
    key_evidence: list[str]         # URLs of key supporting sources
    verdict_score: float
    confidence_label: ConfidenceLevel
    reasons_short: list[str]
    reasons_expert: dict[str, Any]


class LLMScoringOutput(TypedDict, total=False):
    """
    Complete output from LLM scorer.

    LLM produces per-claim verdicts and overall confidence.
    Code validates that confidence respects constraints.
    """
    # Per-claim verdicts
    claim_verdicts: list[ClaimVerdict]

    # Overall scores (0-1)
    overall_confidence: float       # Must be <= constraints.global_cap
    evidence_strength: float        # Aggregate evidence strength
    source_independence: float      # Aggregate source independence
    attribution_integrity: float    # Aggregate attribution integrity

    # Human-readable output
    rationale: str                  # Overall explanation
    evidence_gaps: list[str]        # What evidence is missing?

    # Mapping to existing RGBA (for backward compatibility)
    verified_score: float           # G channel (0-1)
