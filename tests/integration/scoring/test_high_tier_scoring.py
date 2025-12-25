import pytest
import math
from spectrue_core.scoring.priors import calculate_prior
from spectrue_core.scoring.belief import (
    BeliefState, 
    update_belief, 
    process_updates, 
    log_odds_to_prob, 
    calculate_evidence_impact,
    apply_consensus_bound
)
from spectrue_core.schema.scoring import ConsensusState

def test_high_tier_source_scoring():
    """
    US1 Validation: High-tier source with strong evidence must result in high posterior.
    """
    # 1. Prior Calculation (Tier 1 Source)
    # Tier 1 (+1.0 log odds)
    prior_lo = calculate_prior(tier=1, brand_trust=50.0) 
    prior_belief = BeliefState(log_odds=prior_lo)
    
    # Check Initial Belief > 70%
    assert prior_lo > 0.0
    initial_prob = log_odds_to_prob(prior_lo)
    assert initial_prob > 0.7
    
    # 2. Strong Evidence Updates
    # Simulate 3 strong supporting claims
    updates = []
    for _ in range(3):
        impact = calculate_evidence_impact("verified", confidence=0.9)
        updates.append(impact)
        
    final_belief = process_updates(prior_belief, updates)
    
    # 3. Validation
    # 1.0 + 3 * (1.0 * 0.9) = 3.7
    # Prob(3.7) should be very high (> 0.95)
    assert final_belief.log_odds > prior_belief.log_odds
    final_prob = log_odds_to_prob(final_belief.log_odds)
    assert final_prob > 0.95
    
    # Check strict compliance (Must NOT score ~50%)
    assert abs(final_prob - 0.5) > 0.2

def test_consensus_bounding():
    """
    US1 Validation: Consensus affects maximum achievable posterior.
    """
    # Scenario: Strong internal evidence (e.g. log-odds 2.0 / 88%)
    initial = BeliefState(log_odds=2.0) 
    
    # Case A: Consensus is Split (0.5 score)
    # Limit should be around 0.0 log-odds
    consensus_split = ConsensusState(score=0.5, stability=1.0, source_count=10)
    
    bounded_split = apply_consensus_bound(initial, consensus_split)
    
    # Should be capped at 0.0
    assert math.isclose(bounded_split.log_odds, 0.0, abs_tol=1e-5)
    assert math.isclose(log_odds_to_prob(bounded_split.log_odds), 0.5, abs_tol=1e-5)
    
    # Case B: Consensus is Strong Support (0.9 score)
    # Limit should be > 2.0
    consensus_high = ConsensusState(score=0.9, stability=1.0, source_count=10)
    
    bounded_high = apply_consensus_bound(initial, consensus_high)
    
    # Should NOT be capped (original 2.0 preserved)
    assert bounded_high.log_odds == 2.0
