import pytest
import math
from spectrue_core.scoring.belief import sigmoid_impact, calculate_evidence_impact

def test_sigmoid_impact_saturation():
    # Weak strength (0.1) should have near-zero impact
    # Sigmoid steepness k=10, x0=0.5
    weak = sigmoid_impact(strength=0.1, relevance=1.0, direction=1.0)
    assert weak < 0.1
    assert weak > 0.0

    # Strong strength (0.9) should have high impact (approaching l_max=2.0)
    strong = sigmoid_impact(strength=0.9, relevance=1.0, direction=1.0)
    assert strong > 1.8 
    
    # Midpoint (0.5) should be half max (l_max/2 = 1.0)
    # Sigmoid(0) = 0.5. impact = 1.0 * 2.0 * 0.5 = 1.0.
    mid = sigmoid_impact(strength=0.5, relevance=1.0, direction=1.0)
    assert math.isclose(mid, 1.0, abs_tol=1e-5)

def test_sigmoid_relevance_scaling():
    # High strength but low relevance
    impact = sigmoid_impact(strength=0.9, relevance=0.1, direction=1.0)
    # Full impact ~1.96. Scaled by 0.1 -> ~0.196
    assert impact < 0.25
    assert impact > 0.15

def test_calculate_evidence_impact_integration():
    # Verified, Conf=0.9 -> Strong Positive
    res = calculate_evidence_impact("verified", confidence=0.9)
    assert res > 1.5
    
    # Refuted, Conf=0.9 -> Strong Negative
    res_neg = calculate_evidence_impact("refuted", confidence=0.9)
    assert res_neg < -1.5
    
    # Verified, Conf=0.1 -> Weak Positive
    res_weak = calculate_evidence_impact("verified", confidence=0.1)
    assert res_weak < 0.1
