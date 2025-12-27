# Trace & Debugging Guide

Spectrue Engine emits a structured JSONL trace for every run.

## Key events to look at

- `claims.extracted` / `claims.normalized`: how many claims were produced and their metadata.
- `target_selection.completed`: which claims were chosen for retrieval (targets) vs deferred.
- `retrieval.search.*` and `tavily.extract.*`: external calls and their cost impact.
- `evidence.items.summary`: stance distribution (SUPPORT/REFUTE/CONTEXT) per claim.
- `score_evidence.response`: LLM scoring output per claim.
- `verdict.*`: deterministic post-processing and final per-claim verdict.
- `verdict.explainability_tier_factor`: A adjustment from tier prior factor.

## How to diagnose cost spikes

1. Start from `total_credits` and per-phase costs.
2. Count external calls:
   - number of searches
   - number of extracts
3. Inspect why stop-early did or didn't trigger:
   - check readiness / sufficiency events
4. Verify `target_selection`:
   - too many targets => too many retrieval loops

## Claim isolation invariant

Evidence and scoring are **per-claim**. If you see evidence from other claims influencing a score, inspect:

- evidence item `claim_id`
- evidence filtering by claim in `pipeline_evidence.py`
- readiness checks in `sufficiency.py` / `phase_runner.py`
