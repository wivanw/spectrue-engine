import pytest
from unittest.mock import patch, MagicMock
from spectrue_core.use_cases.claims.deep_judge import judge_claims_independently, ClaimJudgeSkill
from spectrue_core.domain.claims.frame import ClaimFrame, JudgeOutput, RGBAScore
from spectrue_core.utils.trace import Trace

@pytest.mark.asyncio
async def test_deep_judge_decision_impact_emits_events():
    """
    T030: Add integration test for decision-impact logging coverage and signal contribution.
    """
    output = JudgeOutput(
        claim_id="c1",
        rgba=RGBAScore(0.1, -1.0, 0.5, 0.5), # G=-1.0 means nei/unverifiable
        confidence=0.8,
        verdict="nei",
        explanation="test",
        missing_evidence=("Need more data",),
        prior_score=0.9, # very high prior
        prior_reason="Known fact"
    )
    
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
    
    with patch.object(ClaimJudgeSkill, "judge", return_value=output):
        with patch.object(Trace, "event") as mock_trace:
            with patch("spectrue_core.use_cases.claims.deep_judge.logger.warning") as mock_warning:
                results = await judge_claims_independently(
                    claim_frames=[frame],
                    evidence_summaries={},
                    llm_client=MagicMock()
                )

                # Print the errors to see why the judge failed
                assert "c1" not in results[1], f"Judge returned error: {results[1].get('c1')}"
                assert not mock_warning.called
        
        # Verify Trace.event was called for decision_impact
        # It's called for start, complete, and decision_impact
        impact_calls = [c for c in mock_trace.call_args_list if c[0][0] == "decision_impact"]
        
        assert len(impact_calls) == 1
        event_name, event_data = impact_calls[0][0]
        
        assert event_name == "decision_impact"
        assert event_data["claim_id"] == "c1"
        assert event_data["module_name"] == "deep_judge_composition"
        assert event_data["did_change_retrieval"] is False
        assert event_data["did_change_confidence"] is True
        assert event_data["did_change_verdict"] is False
        
        assert "missing_evidence_penalty" in event_data["reason_codes"]
        assert "prior_score_supported_blocked" in event_data["reason_codes"]
        
        # Check that stats changed
        assert event_data["before_snapshot"]["confidence"] == 0.8
        assert event_data["before_snapshot"]["verdict"] == "nei"
        assert event_data["after_snapshot"]["verdict"] == "nei"
        assert event_data["after_snapshot"]["confidence"] == 0.5


@pytest.mark.asyncio
async def test_no_positive_upgrade_from_negative_g():
    output = JudgeOutput(
        claim_id="c1",
        rgba=RGBAScore(0.1, -1.0, 0.5, 0.5),
        confidence=0.62,
        verdict="NEI",
        explanation="test",
        missing_evidence=(),
        prior_score=0.95,
        prior_reason="Known fact",
    )

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

    with patch.object(ClaimJudgeSkill, "judge", return_value=output):
        with patch.object(Trace, "event") as mock_trace:
            results, errors = await judge_claims_independently(
                claim_frames=[frame],
                evidence_summaries={},
                llm_client=MagicMock(),
            )

    assert not errors
    assert results["c1"].verdict == "NEI"

    impact_calls = [c for c in mock_trace.call_args_list if c[0][0] == "decision_impact"]
    assert len(impact_calls) == 1
    _, event_data = impact_calls[0][0]
    assert "prior_score_supported_blocked" in event_data["reason_codes"]
    assert event_data["after_snapshot"]["verdict"] == "NEI"
