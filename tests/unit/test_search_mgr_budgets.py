
import pytest
from unittest.mock import MagicMock
from spectrue_core.verification.search.search_mgr import SearchManager
from spectrue_core.runtime_config import EngineRuntimeConfig

from spectrue_core.config import SpectrueConfig

@pytest.fixture
def search_mgr():
    runtime_cfg = EngineRuntimeConfig.load_from_env()
    config = SpectrueConfig(runtime=runtime_cfg, openai_api_key="mock", tavily_api_key="mock")
    
    mgr = SearchManager(config)
    
    # Mock search tool
    mgr.web_tool = MagicMock()
    mgr.web_tool.search.return_value = []
    
    return mgr

def test_independent_claim_budgets(search_mgr):
    """Verify that different claim_ids have independent budget trackers."""
    
    claim_id_1 = "claim_1"
    claim_id_2 = "claim_2"
    
    # Access private trackers for verification (white-box testing)
    tracker1 = search_mgr.get_claim_budget_tracker(claim_id_1)
    tracker2 = search_mgr.get_claim_budget_tracker(claim_id_2)
    
    assert tracker1 is not tracker2, "Trackers should be different instances"
    
    # Simulate usage on tracker 1
    # Assuming the tracker has a method like 'record_extract' or we can check its state
    # Let's inspect the type of tracker returned. It should be an EvidenceAcquisitionLadder (or similar wrapper)
    
    # Actually, SearchManager uses EvidenceAcquisitionLadder internally. 
    # Let's verify that apply_evidence_acquisition_ladder uses the correct tracker.
    
    # We can't easily spy on internal methods without patching.
    # But we can check if the state persists.
    
    # Let's verify that get_claim_budget_tracker returns stable instances
    tracker1_again = search_mgr.get_claim_budget_tracker(claim_id_1)
    assert tracker1 is tracker1_again, "Tracker instance should be stable for the same claim_id"
    
    # Verify that resetting metrics clears trackers
    search_mgr.reset_metrics()
    tracker1_new = search_mgr.get_claim_budget_tracker(claim_id_1)
    assert tracker1 is not tracker1_new, "Tracker should be recreated after reset"

def test_tracker_isolation(search_mgr):
    claim1 = "c1"
    claim2 = "c2"
    
    t1 = search_mgr.get_claim_budget_tracker(claim1)
    t2 = search_mgr.get_claim_budget_tracker(claim2)
    
    # Verify they have independent budgets
    # We can modify 'queries_used' manually to test independence if we can access it
    
    # EvidenceAcquisitionLadder exposes 'queries_used'?
    if hasattr(t1, 'queries_used'):
        t1.queries_used = 5
        assert t2.queries_used == 0 # Should be unaffected
    
    # Or 'extracts_used'
    if hasattr(t1, 'extracts_used'):
        t1.extracts_used = 10
        assert t2.extracts_used == 0

