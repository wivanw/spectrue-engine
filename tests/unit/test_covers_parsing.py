import pytest
from unittest.mock import MagicMock, AsyncMock
from spectrue_core.verification.pipeline.pipeline_evidence import annotate_evidence_stance, EvidenceFlowInput
from spectrue_core.pipeline.mode import AnalysisMode

@pytest.mark.asyncio
async def test_annotate_evidence_stance_normalizes_covers():
    # Setup
    mock_agent = MagicMock()
    # Mock cluster_evidence to return evidence with raw/messy covers
    raw_evidence = [
        {
            "claim_id": "c1",
            "url": "http://example.com/1",
            "covers": ["Entity ", "TIME", "garbage"],
            "stance": "support"
        },
        {
            "claim_id": "c2",
            "url": "http://example.com/2",
            "covers": "not a list", # Should handle this gracefully
            "stance": "refute"
        },
        {
            "claim_id": "c1",
            "url": "http://example.com/3",
            "covers": ["location", "location"], # Should dedup
            "stance": "support"
        }
    ]
    mock_agent.cluster_evidence = AsyncMock(return_value=raw_evidence)

    inp = EvidenceFlowInput(
        fact="test fact",
        original_fact="test fact",
        lang="en",
        content_lang="en",
        analysis_mode=AnalysisMode.DEEP_V2,
        progress_callback=None
    )

    claims = [
        {"id": "c1", "text": "claim 1", "subject_entities": ["E1"], "metadata": {"time_signals": {"year": "2024"}}},
        {"id": "c2", "text": "claim 2", "subject_entities": ["E2"]}
    ]
    sources = [{"url": "http://example.com/1"}]

    # Execute
    result = await annotate_evidence_stance(
        agent=mock_agent,
        inp=inp,
        claims=claims,
        sources=sources
    )

    # Verify normalization
    # Item 0: "Entity " -> "entity", "TIME" -> "time", "garbage" -> removed
    assert set(result[0]["covers"]) == {"entity", "time"}
    
    # Item 1: "not a list" -> empty list
    assert result[1]["covers"] == []
    
    # Item 2: "location" deduped
    assert result[2]["covers"] == ["location"]
