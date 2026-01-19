
from spectrue_core.scoring.budget_allocation import BudgetState, ExtractBudgetParams


def test_inv_051_hard_caps_are_explicit_and_traceable():
    """INV-051: Hard Caps are Explicit and Traceable.

    When a hard ceiling triggers, the stop reason must:
      - name the cap (e.g., cap:max_extracts)
      - include both used and cap values (used=X cap=Y)
    """

    params = ExtractBudgetParams(min_extracts=0, max_extracts=3)
    state = BudgetState(params=params, extracts_used=3)

    should_continue, reason = state.should_continue_extracting()

    assert should_continue is False
    assert reason.startswith("cap:max_extracts")
    assert "used=3" in reason
    assert "cap=3" in reason
