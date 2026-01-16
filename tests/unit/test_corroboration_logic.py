import pytest
from spectrue_core.verification.evidence.dedup_fingerprints import (
    simhash64,
    simhash_bucket_id
)
from spectrue_core.pipeline.steps.evidence_dedup import EvidenceDedupStep
from spectrue_core.pipeline.steps.evidence_corroboration import EvidenceCorroborationStep
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import AnalysisMode
from unittest.mock import MagicMock

def test_fingerprint_logic():
    t1 = "The quick brown fox jumps over the lazy dog."
    t2 = "The quick brown fox jumps over the lazy dog!" # punctuation/case diff
    t3 = "A completely different sentence."

    from spectrue_core.verification.evidence.dedup_fingerprints import normalize_text_for_hash
    n1 = normalize_text_for_hash(t1)
    n2 = normalize_text_for_hash(t2)
    assert n1 == n2

    s1 = simhash64(t1)
    s2 = simhash64(t2)
    s3 = simhash64(t3)
    
    assert s1 == s2 # Should be identical after normalization
    assert s1 != s3
    
    b1 = simhash_bucket_id(s1)
    b2 = simhash_bucket_id(s2)
    assert b1 == b2

@pytest.mark.asyncio
async def test_dedup_and_corroboration_pipeline():
    mode = MagicMock()
    mode.api_analysis_mode = AnalysisMode.DEEP_V2
    
    # Sources with overlapping domains and near-dup content
    sources = [
        {
            "claim_id": "c1",
            "domain": "www.cnn.com",
            "url": "url1",
            "excerpt": "Ignore this",
            "stance": "SUPPORT",
            "quote": "The CEO resigned today." 
        },
        {
            "claim_id": "c1",
            "domain": "cnn.com", # same publisher
            "url": "url2",
            "excerpt": "Ignore this too",
            "stance": "SUPPORT",
            "quote": "The CEO resigned today!" # near dup
        },
        {
            "claim_id": "c1",
            "domain": "bbc.com",
            "url": "url3",
            "excerpt": "A new CEO will be appointed.",
            "stance": "MENTION"
        }
    ]
    claims = [{"id": "c1"}]
    
    ctx = PipelineContext(mode=mode, claims=claims, sources=sources)
    
    # 1. Dedup
    dedup_step = EvidenceDedupStep()
    ctx = await dedup_step.run(ctx)
    
    for s in ctx.sources:
        assert "publisher_id" in s
        assert "content_hash" in s
        assert "similar_cluster_id" in s
    
    assert ctx.sources[0]["publisher_id"] == "cnn.com"
    assert ctx.sources[1]["publisher_id"] == "cnn.com"
    assert ctx.sources[0]["similar_cluster_id"] == ctx.sources[1]["similar_cluster_id"]
    
    # 2. Corroboration
    corr_step = EvidenceCorroborationStep()
    ctx = await corr_step.run(ctx)
    
    corr_by_claim = ctx.get_extra("corroboration_by_claim")
    assert "c1" in corr_by_claim
    c1 = corr_by_claim["c1"]
    
    # 1 precise publisher (cnn.com, because only src[0] has a quote)
    assert c1["precision_publishers_support"] == 1
    # 1 corroboration cluster (cnn items merged)
    assert c1["corroboration_clusters_support"] == 1
    # 2 unique publishers total (cnn, bbc)
    assert c1["unique_publishers_total"] == 2
