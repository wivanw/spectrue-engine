"""Claim processing modules."""

from .claim_selection import pick_ui_main_claim
from .claim_utility import build_claim_utility_features, score_claim_utility
from .coverage_anchors import (
    Anchor,
    AnchorKind,
    extract_all_anchors,
    extract_time_anchors,
    extract_numeric_anchors,
    extract_quote_anchors,
    get_anchor_ids,
    anchors_to_prompt_context,
)
from .coverage_validator import (
    CoverageGap,
    validate_coverage,
    build_gapfill_prompt,
    merge_gapfill_result,
    check_remaining_gaps,
    emit_coverage_summary,
)

__all__ = [
    "pick_ui_main_claim",
    "build_claim_utility_features",
    "score_claim_utility",
    # Coverage anchors
    "Anchor",
    "AnchorKind",
    "extract_all_anchors",
    "extract_time_anchors",
    "extract_numeric_anchors",
    "extract_quote_anchors",
    "get_anchor_ids",
    "anchors_to_prompt_context",
    # Coverage validation
    "CoverageGap",
    "validate_coverage",
    "build_gapfill_prompt",
    "merge_gapfill_result",
    "check_remaining_gaps",
    "emit_coverage_summary",
]
