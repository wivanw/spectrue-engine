
import pytest
from unittest.mock import MagicMock, AsyncMock

from spectrue_core.pipeline.steps.transferred_stance_annotate import (
    TransferredStanceAnnotateStep,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import AnalysisMode

@pytest.mark.asyncio
async def test_transferred_stance_annotate_overwrites_signature():
    # Setup
    agent = MagicMock()
    # Mock annotate_evidence_stance (which we imported in the step)
    # Actually we need to mock it in the module or mock the agent's behavior if it's called through agent
    # But wait, TransferredStanceAnnotateStep calls annotate_evidence_stance(agent=self.agent, ...)
    
    # We can use patch to mock annotate_evidence_stance
    with MagicMock():
        import spectrue_core.pipeline.steps.transferred_stance_annotate as step_module
        step_module.annotate_evidence_stance = AsyncMock(return_value=[
            {"claim_id": "target", "url": "http://example.com/1", "stance": "SUPPORT"}
        ])
        
        target_claim = {
            "id": "target",
            "subject_entities": ["Target Entity"],
            "metadata": {"time_signals": {"time_bucket": "2025"}}
        }
        
        sources = [
            {
                "claim_id": "target",
                "url": "http://example.com/1",
                "provenance": "transferred",
                "origin_claim_id": "origin",
                "event_signature": {"entities": ["Origin Entity"], "time_bucket": "2024"} # Old signature
            }
        ]
        
        mode = MagicMock()
        mode.api_analysis_mode = AnalysisMode.DEEP_V2
        
        ctx = PipelineContext(mode=mode, claims=[target_claim], sources=sources)
        
        config = MagicMock()
        config.runtime.deep_v2.restace_transferred_top_k = 2
        
        step = TransferredStanceAnnotateStep(agent=agent, config=config)
        
        # Execute
        result_ctx = await step.run(ctx)
        
        # Verify
        updated_sources = result_ctx.sources
        assert len(updated_sources) == 1
        ev = updated_sources[0]
        assert ev["provenance"] == "transferred"
        
        # Event signature MUST be updated to target claim's metadata
        assert ev["event_signature"]["entities"] == ["Target Entity"]
        assert ev["event_signature"]["time_bucket"] == "2025"
        
        # Also check provenance markers were preserved
        assert ev["origin_claim_id"] == "origin"
