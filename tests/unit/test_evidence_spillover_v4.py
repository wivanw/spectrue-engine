
from typing import Any
import pytest
from unittest.mock import MagicMock

from spectrue_core.pipeline.steps.evidence_spillover import (
    EvidenceSpilloverStep,
    _claim_topic_signature,
    _topic_overlap_boost,
)
from spectrue_core.pipeline.core import PipelineContext, PipelineMode
from spectrue_core.pipeline.mode import AnalysisMode

def test_claim_topic_signature_extraction():
    claim = {
        "topic_group": "politics",
        "topic_key": "election_2024",
        "metadata": {
            "topic_tags": ["voting", "democracy"]
        },
        "subject_entities": ["Candidate A"],
        "retrieval_seed_terms": ["election fraud"]
    }
    
    sig = _claim_topic_signature(claim)
    
    expected = {
        "topic_group:politics",
        "topic_key:election_2024",
        "voting",
        "democracy",
        "ent:Candidate A",
        "seed:election fraud"
    }
    
    assert sig == expected
    assert isinstance(sig, set)

def test_topic_overlap_boost_calculation():
    c1 = {"topic_group": "science", "subject_entities": ["Mars"]}
    c2 = {"topic_group": "science", "subject_entities": ["Venus"]} # Shares group
    c3 = {"topic_group": "cooking", "subject_entities": ["Mars"]} # Shares entity
    c4 = {"topic_group": "sports"} # No overlap
    
    # 1 overlap (topic_group) -> 0.02
    assert _topic_overlap_boost(c1, c2) == 0.02
    
    # 1 overlap (entity) -> 0.02
    assert _topic_overlap_boost(c1, c3) == 0.02
    
    # 0 overlap -> 0.0
    assert _topic_overlap_boost(c1, c4) == 0.0
    
    # Max cap test
    c_heavy_1 = {"metadata": {"topic_tags": [f"t{i}" for i in range(10)]}}
    c_heavy_2 = {"metadata": {"topic_tags": [f"t{i}" for i in range(10)]}}
    # 10 overlaps * 0.02 = 0.20, should cap at 0.10
    assert _topic_overlap_boost(c_heavy_1, c_heavy_2) == 0.10

@pytest.mark.asyncio
async def test_evidence_spillover_ranking_with_boost():
    # Setup context
    mode = MagicMock(spec=PipelineMode)
    mode.api_analysis_mode = AnalysisMode.DEEP_V2
    
    # Claims
    target_claim = {
        "id": "target",
        "topic_group": "tech",
        "verification_target": "none"
    }
    
    # Origin 1: Matching topic (tech)
    origin_match = {
        "id": "match",
        "topic_group": "tech",
        "verification_target": "none"
    }
    
    # Origin 2: Non-matching topic (food)
    origin_mismatch = {
        "id": "mismatch",
        "topic_group": "food",
        "verification_target": "none"
    }
    
    claims = [target_claim, origin_match, origin_mismatch]
    
    # Sources: Identical relevance/quality, only topic differs
    sources = [
        {
            "claim_id": "match",
            "url": "http://match.com",
            "relevance_score": 0.8,
            "stance": "SUPPORT",
            "quote": "foo",
            "assertion_key": "", # No slot check
        },
        {
            "claim_id": "mismatch",
            "url": "http://mismatch.com",
            "relevance_score": 0.8,
            "stance": "SUPPORT",
            "quote": "bar",
            "assertion_key": "", # No slot check
        }
    ]
    
    ctx = PipelineContext(
        mode=mode,
        claims=claims,
        sources=sources
    )
    # Force cluster map so all are peers
    ctx = ctx.set_extra("cluster_map", {
        "target": "cluster1",
        "match": "cluster1",
        "mismatch": "cluster1"
    })
    
    # Config mock
    config = MagicMock()
    config.runtime.deep_v2.corroboration_top_k = 2  # Get both
    
    step = EvidenceSpilloverStep(config=config)
    
    # Execute
    result_ctx = await step.run(ctx)
    
    # Verify sources
    # We expect 2 original sources + 4 transferred sources
    # (match->target, mismatch->target, mismatch->match, match->mismatch)
    final_sources = result_ctx.sources
    assert len(final_sources) == 6
    
    transferred = [s for s in final_sources if s.get("provenance") == "transferred"]
    assert len(transferred) == 4
    
    # Let's try top_k = 1 to verify ranking
    config.runtime.deep_v2.corroboration_top_k = 1
    
    # Re-run strict
    ctx2 = PipelineContext(mode=mode, claims=claims, sources=sources)
    ctx2 = ctx2.set_extra("cluster_map", {"target": "c1", "match": "c1", "mismatch": "c1"})
    result_ctx2 = await step.run(ctx2)
    
    # Only target MUST have transferred evidence (due to top_k=1 and peers)
    # Actually EVERY claim will have 1 transferred.
    # We check the one transferred to 'target'.
    transferred_to_target = [
        s for s in result_ctx2.sources 
        if s.get("provenance") == "transferred" and s.get("claim_id") == "target"
    ]
    assert len(transferred_to_target) == 1
    
    # The chosen one MUST be from 'match' due to topic boost
    # Base scores are equal (0.8 + 0.05 + 0.05 = 0.9)
    # Boost adds 0.02 to 'match' -> 0.92 vs 0.90
    assert transferred_to_target[0]["origin_claim_id"] == "match"
