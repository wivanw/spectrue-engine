import pytest
from spectrue_core.scoring.priors import calculate_prior

def test_calculate_prior_tier_1_neutral_brand():
    # Tier 1 (+1.0), Brand 50 (Neutral) -> +1.0
    # Expected: log-odds 1.0 (approx 73%)
    prior = calculate_prior(1, 50.0)
    assert prior == 1.0

def test_calculate_prior_tier_4_high_trust():
    # Tier 4 (-1.0), Brand 100 (+1.0 modifier) -> 0.0
    # Trust cancels out low tier
    prior = calculate_prior(4, 100.0)
    assert prior == 0.0

def test_calculate_prior_unknown_tier():
    # Unknown Tier -> Defaults to Tier 3 (0.0)
    # Brand 50 -> 0.0 modifier
    prior = calculate_prior(99, 50.0)
    assert prior == 0.0

def test_calculate_prior_tier_2_low_trust():
    # Tier 2 (+0.5), Brand 0 (-1.0 modifier) -> -0.5
    prior = calculate_prior(2, 0.0)
    assert prior == -0.5
