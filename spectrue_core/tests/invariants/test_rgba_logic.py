import math
import pytest
from spectrue_core.verification.evidence.evidence_alpha import weight_func, compute_A_det, sigmoid

def test_weight_func_basic():
    # r <= 0.5 should have 0 weight
    assert weight_func(0.5) == 0.0
    assert weight_func(0.3) == 0.0
    
    # r > 0.5 should have positive weight
    w_06 = weight_func(0.6)
    assert w_06 > 0
    assert math.isclose(w_06, math.log(0.6/0.4))

def test_alpha_not_decreased_by_weak_support():
    """M119: Adding a low-tier SUPPORT quote should not reduce A_det."""
    items_initial = [
        {"claim_id": "c1", "stance": "SUPPORT", "quote": "Strong evidence", "r_eff": 0.90}
    ]
    a_initial = compute_A_det(items_initial, "c1")
    
    items_with_weak = items_initial + [
        {"claim_id": "c1", "stance": "SUPPORT", "quote": "Weak evidence", "r_eff": 0.51}
    ]
    a_with_weak = compute_A_det(items_with_weak, "c1")
    
    # A_det should increase (or stay same), never decrease
    assert a_with_weak >= a_initial

def test_alpha_ignores_r_leq_0_5():
    """M119: SUPPORT quotes with r_eff <= 0.5 should not increase A_det."""
    items_baseline = [
        {"claim_id": "c1", "stance": "SUPPORT", "quote": "Base evidence", "r_eff": 0.70}
    ]
    a_baseline = compute_A_det(items_baseline, "c1")
    
    items_with_useless = items_baseline + [
        {"claim_id": "c1", "stance": "SUPPORT", "quote": "Useless evidence", "r_eff": 0.50}
    ]
    a_with_useless = compute_A_det(items_with_useless, "c1")
    
    assert math.isclose(a_baseline, a_with_useless)

# M133: compute_alpha_cap removed — LLM A-score passes through unchanged

def test_sigmoid_robustness():
    assert sigmoid(1000) == 1.0
    assert sigmoid(-1000) == 0.0
    assert math.isclose(sigmoid(0), 0.5)

def test_A_det_works_without_support_label():
    """M119 Fix: A_det should work with quoted items even if stance is missing or CONTEXT."""
    items = [
        {"claim_id": "c1", "stance": "CONTEXT", "quote": "Direct evidence", "r_eff": 0.90}
    ]
    a_val = compute_A_det(items, "c1")
    # sigmoid(w(0.9)) = sigmoid(ln(9)) = sigmoid(2.19) ~= 0.9
    assert a_val > 0.8

def test_A_det_ignores_irrelevant():
    """M119 Fix: IRRELEVANT items should not contribute to A_det even if they have quotes."""
    items = [
        {"claim_id": "c1", "stance": "IRRELEVANT", "quote": "Some quote", "r_eff": 0.95}
    ]
    a_val = compute_A_det(items, "c1")
    assert a_val == 0.5 # sigmoid(0)

def test_A_det_increases_with_multiple_anchors():
    """M119: More anchors should increase A_det."""
    items_1 = [{"claim_id": "c1", "stance": "SUPPORT", "quote": "q1", "r_eff": 0.8}]
    items_2 = items_1 + [{"claim_id": "c1", "stance": "REFUTE", "quote": "q2", "r_eff": 0.8}]
    
    a1 = compute_A_det(items_1, "c1")
    a2 = compute_A_det(items_2, "c1")
    
    assert a2 > a1
