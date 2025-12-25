import pytest
import math
from spectrue_core.scoring.belief import prob_to_log_odds, log_odds_to_prob, update_belief
from spectrue_core.schema.scoring import BeliefState

def test_prob_to_log_odds():
    assert math.isclose(prob_to_log_odds(0.5), 0.0, abs_tol=1e-9)
    assert prob_to_log_odds(0.9) > 0
    assert prob_to_log_odds(0.1) < 0
    # Test clipping behavior for 0.0 and 1.0
    # Expected: log((1-epsilon)/epsilon) roughly 20.7 for epsilon=1e-9
    assert prob_to_log_odds(1.0) > 20.0 
    assert prob_to_log_odds(0.0) < -20.0

def test_log_odds_to_prob():
    assert math.isclose(log_odds_to_prob(0.0), 0.5, abs_tol=1e-9)
    assert math.isclose(log_odds_to_prob(100.0), 1.0, abs_tol=1e-9)
    assert math.isclose(log_odds_to_prob(-100.0), 0.0, abs_tol=1e-9)

def test_belief_update():
    initial = BeliefState(log_odds=0.0) # 50%
    evidence_strong_pos = 1.0 
    updated = update_belief(initial, evidence_strong_pos)
    
    assert updated.log_odds == 1.0
    assert updated.probability > 0.5
    assert math.isclose(updated.probability, 1/(1+math.exp(-1.0)), abs_tol=1e-5)

    evidence_strong_neg = -2.0
    updated_again = update_belief(updated, evidence_strong_neg)
    
    # 1.0 - 2.0 = -1.0
    assert updated_again.log_odds == -1.0
    assert updated_again.probability < 0.5
