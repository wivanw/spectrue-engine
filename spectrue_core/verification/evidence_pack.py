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

from typing import Literal, TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# Claim Types
# ─────────────────────────────────────────────────────────────────────────────

ClaimType = Literal["core", "numeric", "timeline", "attribution", "sidefact"]
"""
Claim types for multi-claim extraction:
- core: Main factual assertion
- numeric: Claims with specific numbers/statistics
- timeline: Claims about dates, deadlines, sequences
- attribution: Claims about who said/did something
- sidefact: Secondary supporting facts
"""

Stance = Literal["support", "contradict", "neutral", "unclear"]
"""Position of a source relative to a claim."""

SourceType = Literal[
    "primary",        # Original source (official website, press release)
    "official",       # Official statement (gov, company, org)
    "independent_media",  # Independent news outlet
    "aggregator",     # News aggregator, syndication
    "social",         # Social media post
    "fact_check",     # M63: Fact-check from Oracle (Google Fact Check API)
    "unknown",        # Cannot determine
]
"""Classification of source authority."""


# ─────────────────────────────────────────────────────────────────────────────
# M63: Oracle Check Result
# ─────────────────────────────────────────────────────────────────────────────

OracleStatus = Literal["CONFIRMED", "REFUTED", "MIXED", "EMPTY"]
"""
Oracle verdict status:
- CONFIRMED: Fact-check confirms the claim is true
- REFUTED: Fact-check says the claim is false/fake
- MIXED: Fact-check says partially true or needs context
- EMPTY: No relevant fact-check found
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


class OracleCheckResult(TypedDict, total=False):
    """
    M63: Result from Google Fact Check API with LLM semantic validation.
    
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

class EvidenceRequirement(TypedDict, total=False):
    """What evidence is required to verify this claim."""
    needs_primary_source: bool      # Needs official/primary confirmation
    needs_independent_2x: bool      # Needs 2+ independent sources
    needs_exact_quote: bool         # Needs verbatim quote match
    needs_recent_source: bool       # Needs source from last N days
    max_age_days: int | None        # Max acceptable source age


class Claim(TypedDict, total=False):
    """A single atomic claim extracted from the article."""
    id: str                         # Unique claim ID (c1, c2, ...)
    text: str                       # The claim text (may contain pronouns)
    type: ClaimType                 # Claim classification
    importance: float               # 0-1, how critical to main thesis
    evidence_requirement: EvidenceRequirement
    search_queries: list[str]       # Generated queries for this claim
    check_oracle: bool              # T10: Should this specific claim be checked against Oracle?
    # M62: Context-aware atomization fields
    normalized_text: str            # Self-sufficient statement with pronouns resolved
    topic_group: str                # Topic tag (e.g., "Economy", "War", "Science")
    check_worthiness: float         # 0-1, how important to verify (filters opinions)


# ─────────────────────────────────────────────────────────────────────────────
# Search Result Structure
# ─────────────────────────────────────────────────────────────────────────────

class SearchResult(TypedDict, total=False):
    """A single search result with stance and quality annotations."""
    claim_id: str                   # Which claim this result pertains to
    url: str
    domain: str                     # Registrable domain (example.com)
    title: str
    snippet: str                    # Search result snippet
    content_excerpt: str | None     # Extracted content (if fetched)
    published_at: str | None
    source_type: SourceType
    stance: Stance                  # Position relative to claim
    relevance_score: float          # 0-1, how relevant to claim
    key_snippet: str | None         # Most relevant quote from content
    quote_matches: list[str]        # Exact quotes that match claim
    is_trusted: bool                # From trusted sources registry
    is_duplicate: bool              # Content duplicate of another result
    duplicate_of: str | None        # URL of original if duplicate


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
    # M62: Additional claim metadata for scoring
    topic_group: str | None         # Topic category from claim extraction
    claim_type: str | None          # Claim type (core, numeric, etc.)


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

    # Evidence
    search_results: list[SearchResult]

    # Metrics (code-computed)
    metrics: EvidenceMetrics

    # Constraints (code-enforced)
    constraints: ConfidenceConstraints


# ─────────────────────────────────────────────────────────────────────────────
# LLM Scoring Output
# ─────────────────────────────────────────────────────────────────────────────

class ClaimVerdict(TypedDict, total=False):
    """Verdict for a single claim from LLM."""
    claim_id: str
    verdict: Literal["verified", "refuted", "unverified", "partially_verified"]
    evidence_strength: float        # 0-1, how well supported
    source_independence: float      # 0-1, are sources independent?
    attribution_integrity: float    # 0-1, is attribution correct?
    confidence: float               # 0-1, overall confidence (must respect cap)
    rationale: str                  # Explanation with source citations
    key_evidence: list[str]         # URLs of key supporting sources


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

