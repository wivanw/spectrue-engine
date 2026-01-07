# M127: Coverage Skeleton Extraction

**Status**: ✅ Completed  
**Date**: 2026-01-07

## Goal
Implement 2-phase claim extraction to ensure comprehensive coverage of verifiable claims.

## Deliverables

### Phase 1 & 2: Coverage Skeleton
- [x] `coverage_skeleton.py` with dataclasses:
  - SkeletonEvent, SkeletonMeasurement, SkeletonQuote, SkeletonPolicy
  - CoverageSkeleton, CoverageAnalysis, SkeletonClaimResult
- [x] Regex-based coverage analyzers:
  - `extract_time_mentions_count()` 
  - `extract_number_mentions_count()`
  - `detect_quote_spans_count()`
- [x] Coverage validation with tolerance threshold
- [x] Trace events: claims.skeleton.created, claims.coverage.warning

### Phase 3: Skeleton → Claims
- [x] `skeleton_to_claims()` converter
- [x] Per-type converters (event, measurement, quote, policy)
- [x] `skeleton_item_id` traceability
- [x] Trace event: claims.skeleton.to_claims, claim.dropped

### Schema & Prompts
- [x] `COVERAGE_SKELETON_SCHEMA` in llm_schemas.py
- [x] `build_skeleton_extraction_prompt()` in claims_prompts.py

### Tests
- [x] 35 unit tests for coverage_skeleton module

## Commits
- `71f55e2`: feat(M127): add coverage skeleton extraction phase 1
- `7ecae91`: feat(M127): add skeleton_to_claims() converter (Phase 3)
- `2b86222`: chore: fix ruff lint errors
