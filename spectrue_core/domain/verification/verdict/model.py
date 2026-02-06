from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Literal, Awaitable, Callable


# --- Base Verification Types ---

ProgressCallback = Callable[[str, float, dict | None], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class EvidenceFlowInput:
    fact: str
    original_fact: str
    lang: str
    content_lang: str | None
    analysis_mode: AnalysisMode
    progress_callback: ProgressCallback | None
    prior_belief: BeliefState | None = None
    context_graph: Any | None = None
    claim_extraction_text: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceCollection:
    """Output of evidence collection prior to judging."""
    pack: Any
    claims: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    claim_text_map: dict[str, str]
    anchor_claim: dict[str, Any] | None
    anchor_claim_id: str | None
    time_windows: dict[str, Any]
    current_cost: float


def resolve_stance_pass_mode(profile_name: SearchProfileName | str) -> StancePassMode:
    """Resolve stance pass mode from profile name."""
    if isinstance(profile_name, SearchProfileName):
        normalized = profile_name.value
    else:
        normalized = (str(profile_name) or SearchProfileName.GENERAL.value).strip().lower()

    if normalized == SearchProfileName.DEEP.value:
        return StancePassMode.TWO_PASS
    return StancePassMode.SINGLE


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
    DEEP = "deep"


class SearchDepth(str, Enum):
    """Search depth levels for retrieval."""
    BASIC = "basic"
    ADVANCED = "advanced"


class StancePassMode(str, Enum):
    """Stance detection pass modes."""
    SINGLE = "single"
    TWO_PASS = "two_pass"


@dataclass
class BeliefState:
    log_odds: float
    confidence: float = 0.0

    @property
    def probability(self) -> float:
        return log_odds_to_prob(self.log_odds)


@dataclass
class ConsensusState:
    score: float
    stability: float
    source_count: int


def prob_to_log_odds(p: float, epsilon: float = 1e-9) -> float:
    """Convert probability to log-odds (logit function)."""
    p = max(epsilon, min(1.0 - epsilon, p))
    import math
    return math.log(p / (1.0 - p))


def log_odds_to_prob(log_odds: float) -> float:
    """Convert log-odds to probability (logistic function)."""
    import math
    try:
        return 1.0 / (1.0 + math.exp(-log_odds))
    except OverflowError:
        return 0.0 if log_odds < 0 else 1.0


class RGBAStatus(str, Enum):
    OK = "OK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    UNVERIFIABLE_BY_NATURE = "UNVERIFIABLE_BY_NATURE"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    EVIDENCE_MISMATCH = "EVIDENCE_MISMATCH"
    """Evidence retrieved is non-empty but not about the claim topic (off-topic)."""


@dataclass
class RGBAMetric:
    status: RGBAStatus
    value: float | None = None
    confidence: float | None = None
    uncertainty: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class RGBAResult:
    R: RGBAMetric
    G: RGBAMetric
    B: RGBAMetric
    A: RGBAMetric
    global_reasons: list[str] = field(default_factory=list)
    summary_trace: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class SourceCluster:
    cluster_id: str
    source_ids: list[str]
    representative_source_id: str
    size: int


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


@dataclass
class AssertionVerdict:
    """Verdict for a single assertion."""
    assertion_key: str
    dimension: str = "FACT"
    status: VerdictStatus = VerdictStatus.AMBIGUOUS
    score: float = 0.5
    evidence_count: int = 0
    supporting_urls: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ClaimVerdict:
    """Verdict for a claim (aggregated from assertion verdicts)."""
    claim_id: str
    status: VerdictStatus = VerdictStatus.AMBIGUOUS
    verdict: VerdictStatus = VerdictStatus.AMBIGUOUS
    verdict_state: VerdictState = VerdictState.INSUFFICIENT_EVIDENCE
    verdict_score: float = 0.5
    confidence: str = "low"
    reasons_short: list[str] = field(default_factory=list)
    reasons_expert: dict[str, Any] = field(default_factory=dict)
    assertion_verdicts: list[AssertionVerdict] = field(default_factory=list)
    evidence_count: int = 0
    fact_assertions_verified: int = 0
    fact_assertions_total: int = 0
    reason: str = ""
    key_evidence: list[str] = field(default_factory=list)
    prior_score: float = -1.0
    prior_reason: str = ""


# --- Judge Normalization Logic ---

_STANCE_CANON = {
    "SUPPORT": "SUPPORT",
    "SUPPORTED": "SUPPORT",
    "TRUE": "SUPPORT",
    "CONFIRMED": "SUPPORT",
    "REFUTE": "REFUTE",
    "REFUTED": "REFUTE",
    "FALSE": "REFUTE",
    "DEBUNKED": "REFUTE",
    "MIXED": "MIXED",
    "PARTIAL": "MIXED",
    "PARTLY_TRUE": "MIXED",
    "PARTLY_FALSE": "MIXED",
    "PARTIALLY": "MIXED",
    "NEI": "NEI",
    "UNKNOWN": "NEI",
    "UNVERIFIABLE": "NEI",
    "INSUFFICIENT": "NEI",
    "NOT_ENOUGH_INFO": "NEI",
    "UNCONFIRMED": "NEI",
    "PLAUSIBLE": "NEI",
    "UNCLEAR": "NEI",
}


def normalize_verdict_enum(value: Any) -> str:
    """Normalize judge verdict/stance labels to canonical enums."""
    if value is None:
        return "NEI"
    s = str(value).strip().upper()
    if not s:
        return "NEI"
    return _STANCE_CANON.get(s, "NEI")


def clamp_unit(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    import math
    if v != v:  # NaN
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def normalize_rgba(rgba: Any) -> list[float] | None:
    """Normalize RGBA into [R,G,B,A]."""
    if not isinstance(rgba, list) or len(rgba) != 4:
        return None
    try:
        r = float(rgba[0])
        g = float(rgba[1])
        b = float(rgba[2])
        a = float(rgba[3])
    except Exception:
        return None

    r = clamp_unit(r, 0.0)
    b = clamp_unit(b, 0.0)
    a = clamp_unit(a, 0.0)

    if not (g == -1.0 or (0.0 <= g <= 1.0)):
        g = -1.0

    return [r, g, b, a]


def sanitize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload or {})
    for key in ("verdict", "stance", "label"):
        if key in out:
            out[key] = normalize_verdict_enum(out.get(key))

    rgba = out.get("rgba")
    nr = normalize_rgba(rgba)
    if nr is not None:
        out["rgba"] = nr

    return out


@dataclass
class StructuredDebug:
    """Debug information (not exposed to users)."""
    per_claim: dict[str, Any] = field(default_factory=dict)
    dropped_evidence: list[dict[str, Any]] = field(default_factory=list)
    content_unavailable_count: int = 0
    processing_notes: list[str] = field(default_factory=list)


@dataclass
class StructuredVerdict:
    """Complete verdict output from scoring."""
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    verified_score: float = -1.0
    explainability_score: float = -1.0
    danger_score: float = -1.0
    style_score: float = -1.0
    rationale: str = ""
    structured_debug: StructuredDebug | None = None
    overall_confidence: float = -1.0
    evidence_gaps: list[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if all required scores are present (not sentinel)."""
        return all([
            self.verified_score >= 0,
            self.explainability_score >= 0,
            self.danger_score >= 0,
            self.style_score >= 0,
        ])

    def get_fact_verification_ratio(self) -> tuple[int, int]:
        """Get (verified_count, total_count) for FACT assertions."""
        verified = sum(cv.fact_assertions_verified for cv in self.claim_verdicts)
        total = sum(cv.fact_assertions_total for cv in self.claim_verdicts)
        return verified, total
