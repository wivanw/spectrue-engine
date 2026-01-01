"""Evidence processing modules."""
]
    "enrich_all_claim_verdicts",
    "process_claim_verdicts",
    "enrich_claim_sources",
    "assign_claim_rgba",
    "check_has_direct_evidence",
    "detect_evidence_conflict",
    "derive_verdict_from_score",
    "derive_verdict_state_from_llm_score",
    "count_stance_evidence",
    "CANONICAL_VERDICT_STATES",
    "find_best_tier_for_claim",
    "compute_explainability_tier_adjustment",
    "get_tier_rank",
    "mark_anchor_duplicates_async",
    "select_anchor_for_article_g",
    "compute_article_g_from_anchor",
    "explainability_factor_for_tier",
    "TIER_A_BASELINE",
    "sigmoid",
    "logit",
    "is_prob",
    "norm_id",
    "build_evidence_pack",
__all__ = [

)
    enrich_all_claim_verdicts,
    process_claim_verdicts,
from .evidence_verdict_processing import (
)
    enrich_claim_sources,
    assign_claim_rgba,
    check_has_direct_evidence,
    detect_evidence_conflict,
    derive_verdict_from_score,
    derive_verdict_state_from_llm_score,
    count_stance_evidence,
    CANONICAL_VERDICT_STATES,
from .evidence_stance import (
)
    find_best_tier_for_claim,
    compute_explainability_tier_adjustment,
    get_tier_rank,
from .evidence_explainability import (
)
    mark_anchor_duplicates_async,
    select_anchor_for_article_g,
    compute_article_g_from_anchor,
    explainability_factor_for_tier,
    TIER_A_BASELINE,
    sigmoid,
    logit,
    is_prob,
    norm_id,
from .evidence_scoring import (
from .evidence import build_evidence_pack

у