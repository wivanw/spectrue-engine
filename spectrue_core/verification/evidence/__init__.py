"""Evidence processing modules."""

from .evidence import build_evidence_pack
from .evidence_scoring import (
    mark_anchor_duplicates_async,
    select_anchor_for_article_g,
    compute_article_g_from_anchor,
    explainability_factor_for_tier,
    TIER_A_BASELINE,
    sigmoid,
    logit,
    is_prob,
    norm_id,
)
from .evidence_explainability import (
    find_best_tier_for_claim,
    get_tier_rank,
)
from .evidence_stance import (
    enrich_claim_sources,
    assign_claim_rgba,
    check_has_direct_evidence,
    detect_evidence_conflict,
    derive_verdict_from_score,
    derive_verdict_state_from_llm_score,
    count_stance_evidence,
    CANONICAL_VERDICT_STATES,
)
from spectrue_core.verification.evidence_verdict_processing import (
    enrich_all_claim_verdicts,
    process_claim_verdicts,
)
from .bayesian_update import apply_bayesian_update

__all__ = [
    "build_evidence_pack",
    "mark_anchor_duplicates_async",
    "select_anchor_for_article_g",
    "compute_article_g_from_anchor",
    "explainability_factor_for_tier",
    "TIER_A_BASELINE",
    "sigmoid",
    "logit",
    "is_prob",
    "norm_id",
    "find_best_tier_for_claim",
    "get_tier_rank",
    "enrich_claim_sources",
    "assign_claim_rgba",
    "check_has_direct_evidence",
    "detect_evidence_conflict",
    "derive_verdict_from_score",
    "derive_verdict_state_from_llm_score",
    "count_stance_evidence",
    "CANONICAL_VERDICT_STATES",
    "enrich_all_claim_verdicts",
    "process_claim_verdicts",
    "apply_bayesian_update",
]