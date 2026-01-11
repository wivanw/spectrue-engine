import pytest
from unittest.mock import AsyncMock, MagicMock
from spectrue_core.verification.retrieval.cegs_mvp import (
    compute_deficit,
    escalate_claim,
    EvidencePool,
    EvidenceItem,
    EvidenceSourceMeta,
    EvidenceBundle
)

@pytest.fixture
def mock_search_mgr():
    mgr = MagicMock()
    # Return valid structure for search results
    mgr.search_phase = AsyncMock(return_value=("snippet", [{"url": "http://escalation.com", "title": "Escalation", "content": "Found it", "score": 0.9}]))
    mgr.fetch_url_content = AsyncMock(return_value="Full content for escalation")
    mgr.fetch_urls_content_batch = AsyncMock(return_value={"http://escalation.com": "Full content for escalation"})
    return mgr

@pytest.fixture
def empty_bundle():
    return EvidenceBundle(
        claim_id="c1",
        matched_items=[],
        cluster_ids=[],
        coverage_flags={}
    )

@pytest.fixture
def sample_claim():
    return {
        "id": "c1",
        "text": "Escalation needed.",
        "normalized_text": "Escalation needed.",
        "subject": "Escalation",
        "context_entities": ["test"],
        "subject_entities": ["Escalation"],
        "risk_score": 0.8
    }

def test_compute_deficit_no_matches(sample_claim, empty_bundle):
    deficit = compute_deficit(sample_claim, empty_bundle)
    assert deficit.is_deficit is True
    assert "no_pool_matches" in deficit.reason_codes
    assert deficit.severity == 0.8

def test_compute_deficit_low_independence(sample_claim):
    # Bundle with 1 match
    meta = EvidenceSourceMeta(url="u1", title="T", snippet="S", score=1.0)
    item = EvidenceItem(url="u1", extracted_text="T", content_hash="h", source_meta=meta)
    bundle = EvidenceBundle(
        claim_id="c1",
        matched_items=[item],
        cluster_ids=["cluster1"],
        coverage_flags={}
    )
    
    deficit = compute_deficit(sample_claim, bundle)
    # Depending on MVP implementation, 1 match might trigger low_independence
    assert deficit.is_deficit is True
    assert "low_independence" in deficit.reason_codes
    assert deficit.severity == 0.8

@pytest.mark.asyncio
async def test_escalate_claim_success(sample_claim, mock_search_mgr):
    pool = EvidencePool()
    
    # Escalation should run, find evidence, merge to pool, and resolve deficit
    # We mock search_mgr to return good evidence that matches "Escalation" (in sample_claim)
    
    updated_pool, new_bundle, collected_urls = await escalate_claim(sample_claim, pool, mock_search_mgr)
    
    assert len(updated_pool.items) > 0
    assert len(new_bundle.matched_items) > 0
    assert mock_search_mgr.search_phase.called
    assert set(collected_urls) == {"http://escalation.com"}

@pytest.mark.asyncio
async def test_escalate_claim_skip_extraction(sample_claim, mock_search_mgr):
    pool = EvidencePool()
    
    # Run with skip_extraction=True
    updated_pool, new_bundle, collected_urls = await escalate_claim(
        sample_claim, 
        pool, 
        mock_search_mgr, 
        skip_extraction=True
    )
    
    # Should collect URL but NOT fetch content
    assert set(collected_urls) == {"http://escalation.com"}
    assert not mock_search_mgr.fetch_urls_content_batch.called
    assert mock_search_mgr.search_phase.called
    
    # Pool items should have empty text
    assert len(updated_pool.items) > 0
    assert updated_pool.items[0].extracted_text == ""
