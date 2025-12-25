import pytest
from spectrue_core.scoring.belief import BeliefState, process_updates, calculate_evidence_impact

def test_noise_tolerance():
    """
    US2 Validation: Weak evidence should have minimal impact on established belief.
    """
    # 1. Start with Strong Belief
    # log-odds 2.0 (~88%)
    initial = BeliefState(log_odds=2.0)
    
    # 2. Inject 5 weak claims (Low confidence)
    # Simulate "Noise" trying to drag it down (Refuted but weak confidence)
    updates = []
    for _ in range(5):
        # Weak impact (Conf=0.1)
        # With Sigmoid k=10, x0=0.5, l_max=2.0
        # Impact should be very small (~0.036) per item
        impact = calculate_evidence_impact("refuted", confidence=0.1)
        updates.append(impact)
        
    final = process_updates(initial, updates)
    
    # 3. Verify minimal change
    delta = abs(final.log_odds - initial.log_odds)
    
    # 5 items * ~0.036 = ~0.18 total change
    # If linear (Conf 0.1 * Base 1.0) -> 0.1 per item -> 0.5 total.
    # Sigmoid suppresses it significantly.
    
    assert delta < 0.25
    
    # Verify it didn't collapse belief
    # 2.0 - 0.18 = 1.82. Still strong.
    assert final.log_odds > 1.8
