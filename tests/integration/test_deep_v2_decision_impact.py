import pytest
from unittest.mock import patch, MagicMock
from spectrue_core.use_cases.claims.deep_judge import judge_claims_independently, ClaimJudgeSkill
from spectrue_core.domain.claims.frame import ClaimFrame, JudgeOutput, RGBAScore
from spectrue_core.utils.trace import Trace

@pytest.mark.asyncio
async def test_deep_judge_keeps_judge_verdict_and_confidence():
    """No midpoint cap for missing evidence, no verdict flip from prior_score (2026-10-18)."""
    output = JudgeOutput(
        claim_id="c1",
        rgba=RGBAScore(0.1, -1.0, 0.5, 0.5),  # G=-1.0: NEI
        confidence=0.8,
        verdict="nei",
        explanation="test",
        missing_evidence=("Need more data",),
        prior_score=0.9,  # strong prior must not flip NEI to supported
        prior_reason="Known fact",
    )
    frame = _frame()
    with patch.object(ClaimJudgeSkill, "judge", return_value=output):
        with patch.object(Trace, "event") as mock_trace:
            results, errors = await judge_claims_independently(
                claim_frames=[frame], evidence_summaries={}, llm_client=MagicMock())
    assert not errors
    assert results["c1"].verdict == "nei" and results["c1"].confidence == 0.8   # was capped to 0.5
    assert not [c for c in mock_trace.call_args_list if c[0][0] == "decision_impact"]


@pytest.mark.asyncio
async def test_prior_does_not_flip_nei_to_refuted():
    output = JudgeOutput(claim_id="c1", rgba=RGBAScore(0.1, -1.0, 0.5, 0.5), confidence=0.62, verdict="NEI",
                         explanation="test", missing_evidence=(), prior_score=0.1, prior_reason="Known false")
    with patch.object(ClaimJudgeSkill, "judge", return_value=output):
        results, errors = await judge_claims_independently(
            claim_frames=[_frame()], evidence_summaries={}, llm_client=MagicMock())
    assert not errors and results["c1"].verdict == "NEI" and results["c1"].confidence == 0.62


def _frame():
    class MockStats:
        total_sources = 0
        publishers_total = 0
        direct_quotes = 0
        exact_dupes_total = 0
        similar_clusters_total = 0
        conflicting_evidence = False
        missing_sources = True
        missing_direct_quotes = False

        class support:
            precision_publishers = 0
            corroboration_clusters = 0

        class refute:
            precision_publishers = 0
            corroboration_clusters = 0

    frame = MagicMock(spec=ClaimFrame)
    frame.claim_id = "c1"
    frame.claim_text = "The earth is round"
    frame.claim_language = "en"
    frame.evidence_items = ()
    frame.evidence_stats = MockStats()
    frame.confirmation_counts = MagicMock()
    return frame
