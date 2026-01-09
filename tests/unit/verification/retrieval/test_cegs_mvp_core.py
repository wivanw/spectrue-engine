import pytest
from unittest.mock import AsyncMock, MagicMock
from spectrue_core.verification.retrieval.cegs_mvp import (
    build_doc_query_plan,
    doc_retrieve_to_pool,
    match_claim_to_pool,
    EvidencePool,
    EvidenceItem,
    EvidenceSourceMeta
)
from spectrue_core.verification.claims.coverage_anchors import Anchor, AnchorKind

@pytest.fixture
def mock_search_mgr():
    mgr = MagicMock()
    mgr.search_phase = AsyncMock(return_value=("snippet", [{"url": "http://example.com", "title": "Test Title", "content": "Test Snippet", "score": 0.9}]))
    mgr.fetch_url_content = AsyncMock(return_value="Full content with entities")
    return mgr

@pytest.fixture
def sample_claims():
    return [
        {
            "id": "c1",
            "text": "Apple released iPhone 15 in 2023.",
            "normalized_text": "Apple released iPhone 15 in 2023.",
            "subject": "Apple",
            "context_entities": ["iPhone", "smartphone"],
            "subject_entities": ["Apple Inc"]
        },
        {
            "id": "c2",
            "text": "It costs $999.",
            "normalized_text": "It costs $999.",
            "subject": "iPhone 15",
            "context_entities": ["price", "cost"],
            "subject_entities": []
        }
    ]

@pytest.fixture
def sample_anchors():
    return [
        Anchor(anchor_id="t1", kind=AnchorKind.TIME, span_text="2023", char_start=0, char_end=4, context_window=""),
        Anchor(anchor_id="n1", kind=AnchorKind.NUMBER, span_text="15", char_start=0, char_end=2, context_window="")
    ]

def test_build_doc_query_plan_limits(sample_claims, sample_anchors):
    queries = build_doc_query_plan(sample_claims, sample_anchors)
    assert isinstance(queries, list)
    assert len(queries) >= 2
    assert len(queries) <= 4
    for q in queries:
        assert len(q) <= 80

@pytest.mark.asyncio
async def test_doc_retrieve_to_pool(mock_search_mgr):
    sanity_terms = {"test", "title"}
    pool = await doc_retrieve_to_pool(
        doc_queries=["query1"],
        sanity_terms=sanity_terms,
        search_mgr=mock_search_mgr
    )
    
    assert isinstance(pool, EvidencePool)
    assert len(pool.items) > 0
    assert pool.items[0].url == "http://example.com"
    # Verify search was called
    mock_search_mgr.search_phase.assert_called_once()
    mock_search_mgr.fetch_url_content.assert_called_once()

def test_match_claim_to_pool(sample_claims):
    # Setup pool with matching item
    meta = EvidenceSourceMeta(url="u1", title="Apple iPhone 15", snippet="Released in 2023", score=1.0)
    item = EvidenceItem(
        url="u1", 
        extracted_text="Full text about Apple iPhone 15 released in 2023.", 
        content_hash="hash", 
        source_meta=meta
    )
    pool = EvidencePool(items=[item], meta=[meta])
    
    claim = sample_claims[0] # Has "Apple", "iPhone", "2023"
    
    bundle = match_claim_to_pool(claim, pool)
    
    assert bundle.claim_id == "c1"
    assert len(bundle.matched_items) == 1
    assert bundle.matched_items[0].url == "u1"
    assert bundle.coverage_flags["has_time_anchor"] is True

def test_match_claim_to_pool_no_overlap(sample_claims):
    # Setup pool with NO entity overlap at all
    meta = EvidenceSourceMeta(url="u2", title="Unrelated topic", snippet="Something else", score=1.0)
    item = EvidenceItem(
        url="u2", 
        extracted_text="Completely unrelated content about bananas.", 
        content_hash="hash", 
        source_meta=meta
    )
    pool = EvidencePool(items=[item], meta=[meta])
    
    claim = sample_claims[0]  # Has entities "Apple", "iPhone", "smartphone", etc.
    
    bundle = match_claim_to_pool(claim, pool)
    
    # With relaxed matching, still needs some semantic tie (base_overlap > 0)
    assert len(bundle.matched_items) == 0  # No entity overlap
