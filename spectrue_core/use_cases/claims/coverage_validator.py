"""Backward-compatible exports for coverage validation."""

from spectrue_core.domain.claims.extraction import (  # noqa: F401
    CoverageGap,
    validate_coverage,
    build_gapfill_prompt,
    merge_gapfill_result,
    check_remaining_gaps,
    emit_coverage_summary,
)

__all__ = [
    "CoverageGap",
    "validate_coverage",
    "build_gapfill_prompt",
    "merge_gapfill_result",
    "check_remaining_gaps",
    "emit_coverage_summary",
]
