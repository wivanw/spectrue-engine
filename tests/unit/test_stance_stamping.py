
from typing import Any
import pytest
from unittest.mock import MagicMock, AsyncMock

from spectrue_core.verification.pipeline.pipeline_evidence import (
    annotate_evidence_stance,
    EvidenceFlowInput
)
from spectrue_core.pipeline.mode import AnalysisMode

@pytest.mark.asyncio
async def test_annotate_evidence_stance_stamps_signature():
    # Setup
    agent = MagicMock()
    # Mock cluster_evidence to return an item with claim_id
    agent.cluster_evidence = AsyncMock(return_value=[
        {"claim_id": "c1", "url": "http://example.com/1"},
        {"claim_id": "c2", "url": "http://example.com/2"}
    ])
    
    inp = MagicMock(spec=EvidenceFlowInput)
    inp.analysis_mode = AnalysisMode.DEEP_V2
    inp.progress_callback = None
    
    claims = [
        {
            "id": "c1",
            "subject_entities": ["Apple"],
            "metadata": {"time_signals": {"year": "2024"}, "locale_signals": {"country": "USA"}}
        },
        {
            "id": "c2",
            "subject_entities": ["Google"],
            "metadata": {"time_signals": {"year": "2023"}, "locale_signals": {"country": "UK"}}
        }
    ]
    
    sources = [{"url": "http://example.com/1"}, {"url": "http://example.com/2"}]
    
    # Execute
    result = await annotate_evidence_stance(
        agent=agent,
        inp=inp,
        claims=claims,
        sources=sources
    )
    
    # Verify
    assert len(result) == 2
    
    # Check c1 signature
    ev1 = next(x for x in result if x["claim_id"] == "c1")
    assert ev1["event_signature"]["entities"] == ["Apple"]
    assert ev1["event_signature"]["time_bucket"] == "2024"
    assert ev1["event_signature"]["locale"] == "USA"
    
    # Check c2 signature
    ev2 = next(x for x in result if x["claim_id"] == "c2")
    assert ev2["event_signature"]["entities"] == ["Google"]
    assert ev2["event_signature"]["time_bucket"] == "2023"
    assert ev2["event_signature"]["locale"] == "UK"
