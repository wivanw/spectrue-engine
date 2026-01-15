
from typing import Any
import pytest
from unittest.mock import MagicMock

from spectrue_core.pipeline.steps.evidence_spillover import (
    EvidenceSpilloverStep,
)
from spectrue_core.verification.evidence.event_signature import (
    claim_event_signature,
    evidence_event_signature,
    signature_compatible,
    EventSignature
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import AnalysisMode

def test_claim_event_signature_extraction():
    claim = {
        "subject_entities": ["Apple Inc.", "iPhone"],
        "metadata": {
            "time_signals": {"time_bucket": "2024"},
            "locale_signals": {"country": "USA"}
        }
    }
    sig = claim_event_signature(claim)
    assert sig.entities == ("Apple Inc.", "iPhone")
    assert sig.time_bucket == "2024"
    assert sig.locale == "USA"

def test_signature_compatibility():
    s1 = EventSignature(entities=("A",), time_bucket="2024", locale="US")
    s2 = EventSignature(entities=("A",), time_bucket="2023", locale="UK") # Entity overlap
    s3 = EventSignature(entities=("B",), time_bucket="2024", locale="US") # Time+Locale overlap
    s4 = EventSignature(entities=("B",), time_bucket="2023", locale="US") # Only Locale
    s5 = EventSignature(entities=("B",), time_bucket="2024", locale="UK") # Only Time
    s6 = EventSignature(entities=("B",), time_bucket="2023", locale="UK") # No overlap
    
    assert signature_compatible(s1, s2) is True
    assert signature_compatible(s1, s3) is True
    assert signature_compatible(s1, s4) is False
    assert signature_compatible(s1, s5) is False
    assert signature_compatible(s1, s6) is False
    
    # Missing evidence signature (None) should not block
    assert signature_compatible(s1, None) is True

@pytest.mark.asyncio
async def test_evidence_spillover_gated_by_event_signature():
    # Setup
    mode = MagicMock()
    mode.api_analysis_mode = AnalysisMode.DEEP_V2
    
    target_claim = {
        "id": "target",
        "subject_entities": ["Topic A"],
        "metadata": {"time_signals": {"time_bucket": "2024"}},
        "verification_target": "none"
    }
    
    # Origin 1: Compatible signature (entity overlap)
    origin_ok = {
        "id": "origin_ok",
        "subject_entities": ["Topic A"],
        "verification_target": "none"
    }
    
    # Origin 2: Incompatible signature
    origin_bad = {
        "id": "origin_bad",
        "subject_entities": ["Topic B"],
        "metadata": {"time_signals": {"time_bucket": "2023"}},
        "verification_target": "none"
    }
    
    sources = [
        {
            "claim_id": "origin_ok",
            "url": "http://ok.com",
            "relevance_score": 0.8,
            "stance": "SUPPORT",
            "quote": "good",
            "event_signature": {"entities": ["Topic A"], "time_bucket": "2024"}
        },
        {
            "claim_id": "origin_bad",
            "url": "http://bad.com",
            "relevance_score": 0.8,
            "stance": "SUPPORT",
            "quote": "bad",
            "event_signature": {"entities": ["Topic B"], "time_bucket": "2023"}
        }
    ]
    
    ctx = PipelineContext(mode=mode, claims=[target_claim, origin_ok, origin_bad], sources=sources)
    ctx = ctx.set_extra("cluster_map", {"target": "c1", "origin_ok": "c1", "origin_bad": "c1"})
    
    config = MagicMock()
    config.runtime.deep_v2.corroboration_top_k = 10
    
    step = EvidenceSpilloverStep(config=config)
    result = await step.run(ctx)
    
    transferred = [s for s in result.sources if s.get("provenance") == "transferred" and s["claim_id"] == "target"]
    
    # Only ok.com should be transferred to target
    assert len(transferred) == 1
    assert transferred[0]["url"] == "http://ok.com"
