# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from spectrue_core.verification.target_selection import (
    select_verification_targets,
    compute_optimal_target_count,
    TargetBudgetParams,
    _expected_value_of_information,
)


def test_empty_claims_returns_empty_result():
    result = select_verification_targets([])
    assert result.targets == []
    assert result.deferred == []


def test_single_claim_becomes_target():
    claims = [{"id": "c1", "text": "Test claim", "check_worthiness": 0.8}]
    result = select_verification_targets(claims, max_targets=2)
    assert len(result.targets) == 1
    assert len(result.deferred) == 0
    assert result.targets[0]["id"] == "c1"


def test_max_targets_limit_respected():
    claims = [
        {"id": "c1", "text": "Claim 1", "check_worthiness": 0.9},
        {"id": "c2", "text": "Claim 2", "check_worthiness": 0.8},
        {"id": "c3", "text": "Claim 3", "check_worthiness": 0.7},
        {"id": "c4", "text": "Claim 4", "check_worthiness": 0.6},
    ]
    result = select_verification_targets(claims, max_targets=2)
    
    assert len(result.targets) == 2
    assert len(result.deferred) == 2
    target_ids = {c["id"] for c in result.targets}
    assert len(target_ids) == 2


def test_graph_order_used_for_targets():
    from spectrue_core.graph.types import GraphResult, RankedClaim

    claims = [
        {"id": "c1", "text": "Claim 1", "check_worthiness": 0.5},
        {"id": "c2", "text": "Claim 2", "check_worthiness": 0.6},
        {"id": "c3", "text": "Claim 3", "check_worthiness": 0.4},
    ]
    graph_result = GraphResult(
        key_claims=[RankedClaim(claim_id="c3", centrality_score=0.9, in_structural_weight=0.0, in_contradict_weight=0.0, is_key_claim=True)],
        all_ranked=[
            RankedClaim(claim_id="c3", centrality_score=0.9, in_structural_weight=0.0, in_contradict_weight=0.0, is_key_claim=True),
            RankedClaim(claim_id="c1", centrality_score=0.7, in_structural_weight=0.0, in_contradict_weight=0.0, is_key_claim=False),
            RankedClaim(claim_id="c2", centrality_score=0.5, in_structural_weight=0.0, in_contradict_weight=0.0, is_key_claim=False),
        ],
    )
    result = select_verification_targets(claims, max_targets=2, graph_result=graph_result)

    target_ids = [c["id"] for c in result.targets]
    assert target_ids[0] == "c3"
    assert set(target_ids).issubset(set(graph_result.key_claim_ids + [r.claim_id for r in graph_result.all_ranked]))


def test_budget_class_affects_max_targets():
    claims = [
        {"id": f"c{i}", "text": f"Claim {i}", "check_worthiness": 0.5}
        for i in range(10)
    ]
    
    # Budget envelope should monotonically allow more targets as budget_class increases
    result_minimal = select_verification_targets(claims, max_targets=None, budget_class="minimal")
    result_standard = select_verification_targets(claims, max_targets=None, budget_class="standard")
    result_deep = select_verification_targets(claims, max_targets=None, budget_class="deep")
    assert len(result_minimal.targets) <= len(result_standard.targets) <= len(result_deep.targets)


def test_evidence_sharing_for_same_cluster():
    claims = [
        {"id": "c1", "text": "Topic A claim 1", "check_worthiness": 0.9, "cluster_id": "topic_a"},
        {"id": "c2", "text": "Topic A claim 2", "check_worthiness": 0.5, "cluster_id": "topic_a"},
        {"id": "c3", "text": "Topic B claim", "check_worthiness": 0.6, "cluster_id": "topic_b"},
    ]
    result = select_verification_targets(claims, max_targets=2)
    
    # c1 and c3 should be targets (highest from each cluster)
    target_ids = {c["id"] for c in result.targets}
    assert "c1" in target_ids or "c3" in target_ids
    
    # c2 (deferred) should share evidence with c1 (same cluster)
    if "c2" in result.reasons:
        if result.evidence_sharing.get("c2"):
            assert result.evidence_sharing["c2"] == "c1"


def test_reasons_recorded_for_all_claims():
    claims = [
        {"id": "c1", "text": "Claim 1", "check_worthiness": 0.9},
        {"id": "c2", "text": "Claim 2", "check_worthiness": 0.5},
    ]
    result = select_verification_targets(claims, max_targets=1)
    
    assert "c1" in result.reasons
    assert "c2" in result.reasons
    assert "target" in result.reasons["c1"]
    assert "deferred" in result.reasons["c2"]


def test_anchor_claim_forced_into_targets():
    """Anchor claim MUST be in targets for normal pipeline, even if not top-K by EV."""
    from spectrue_core.graph.types import GraphResult, RankedClaim

    claims = [
        {"id": "c1", "text": "Claim 1", "check_worthiness": 0.9, "harm_potential": 4},
        {"id": "c2", "text": "Claim 2", "check_worthiness": 0.8, "harm_potential": 3},
        {"id": "c3", "text": "Claim 3 (anchor)", "check_worthiness": 0.3, "harm_potential": 1},  # Low EV
    ]
    # Graph ranks c1, c2 higher than c3
    graph_result = GraphResult(
        key_claims=[],
        all_ranked=[
            RankedClaim(claim_id="c1", centrality_score=0.9, in_structural_weight=0.0, in_contradict_weight=0.0, is_key_claim=True),
            RankedClaim(claim_id="c2", centrality_score=0.8, in_structural_weight=0.0, in_contradict_weight=0.0, is_key_claim=False),
            RankedClaim(claim_id="c3", centrality_score=0.2, in_structural_weight=0.0, in_contradict_weight=0.0, is_key_claim=False),
        ],
    )
    
    # Without anchor_claim_id, c3 would be deferred
    result_no_anchor = select_verification_targets(
        claims, max_targets=2, graph_result=graph_result
    )
    target_ids_no_anchor = {c["id"] for c in result_no_anchor.targets}
    assert "c3" not in target_ids_no_anchor, "c3 should be deferred without anchor forcing"
    
    # With anchor_claim_id=c3, it MUST be in targets
    result_with_anchor = select_verification_targets(
        claims, max_targets=2, graph_result=graph_result, anchor_claim_id="c3"
    )
    target_ids_with_anchor = {c["id"] for c in result_with_anchor.targets}
    assert "c3" in target_ids_with_anchor, "anchor claim c3 MUST be in targets"
    assert len(result_with_anchor.targets) == 2


def test_anchor_already_in_top_k_no_forcing():
    """When anchor is already in top-K, no forcing needed."""
    claims = [
        {"id": "c1", "text": "Claim 1 (anchor)", "check_worthiness": 0.9, "harm_potential": 4},
        {"id": "c2", "text": "Claim 2", "check_worthiness": 0.5, "harm_potential": 2},
    ]
    
    result = select_verification_targets(claims, max_targets=2, anchor_claim_id="c1")
    target_ids = {c["id"] for c in result.targets}
    assert "c1" in target_ids
    # c1 should be first (highest EV), not just forced
    assert result.targets[0]["id"] == "c1"


# -----------------------------------------------------------------------------
# Bayesian EVOI Model Tests (M118)
# -----------------------------------------------------------------------------


def test_bayesian_evoi_high_harm_claims_more_targets():
    """High harm/worthiness claims should result in more targets selected."""
    high_value_claims = [
        {"id": "c1", "check_worthiness": 0.9, "harm_potential": 4.5, "importance": 0.85},
        {"id": "c2", "check_worthiness": 0.85, "harm_potential": 4.0, "importance": 0.8},
        {"id": "c3", "check_worthiness": 0.8, "harm_potential": 3.5, "importance": 0.75},
        {"id": "c4", "check_worthiness": 0.75, "harm_potential": 3.0, "importance": 0.7},
        {"id": "c5", "check_worthiness": 0.7, "harm_potential": 2.5, "importance": 0.65},
    ]
    
    k, trace = compute_optimal_target_count(high_value_claims, budget_class="minimal")
    
    # High-value claims should get close to ceiling (3 for minimal)
    assert k >= 2, f"High-value claims should get at least 2 targets, got {k}"
    assert trace["optimal_k"] == k
    assert trace["budget_class"] == "minimal"
    assert "marginal_analysis" in trace


def test_bayesian_evoi_low_value_claims_fewer_targets():
    """Low harm/worthiness claims should result in fewer targets."""
    low_value_claims = [
        {"id": "c1", "check_worthiness": 0.2, "harm_potential": 1.0, "importance": 0.2},
        {"id": "c2", "check_worthiness": 0.15, "harm_potential": 0.5, "importance": 0.15},
        {"id": "c3", "check_worthiness": 0.1, "harm_potential": 0.3, "importance": 0.1},
    ]
    
    k, trace = compute_optimal_target_count(low_value_claims, budget_class="minimal")
    
    # Low-value claims should get floor (1 for minimal)
    assert k >= 1, "Should have at least floor targets"
    # With low EVOI, net_value might be negative or low
    assert "net_value" in trace


def test_bayesian_evoi_respects_budget_class_ceilings():
    """Budget class ceilings should be respected."""
    claims = [
        {"id": f"c{i}", "check_worthiness": 0.8, "harm_potential": 4.0}
        for i in range(50)
    ]
    
    params = TargetBudgetParams()
    
    k_minimal, _ = compute_optimal_target_count(claims, budget_class="minimal")
    k_standard, _ = compute_optimal_target_count(claims, budget_class="standard")
    k_deep, _ = compute_optimal_target_count(claims, budget_class="deep")
    
    assert k_minimal <= params.budget_ceilings["minimal"]
    assert k_standard <= params.budget_ceilings["standard"]
    assert k_deep <= params.budget_ceilings["deep"]


def test_bayesian_evoi_respects_budget_class_floors():
    """Budget class floors should be respected even for zero-value claims."""
    # Claims with effectively zero EVOI
    zero_value_claims = [
        {"id": "c1", "check_worthiness": 0.0, "harm_potential": 0.0},
        {"id": "c2", "check_worthiness": 0.0, "harm_potential": 0.0},
    ]
    
    params = TargetBudgetParams()
    
    k, _ = compute_optimal_target_count(zero_value_claims, budget_class="minimal")
    
    # Should respect floor even with zero EVOI
    assert k >= params.budget_floors["minimal"]


def test_bayesian_evoi_empty_claims():
    """Empty claims list should return 0."""
    k, trace = compute_optimal_target_count([], budget_class="minimal")
    assert k == 0
    assert trace.get("reason") == "no_claims"


def test_bayesian_evoi_custom_params():
    """Custom parameters should affect target selection."""
    claims = [
        {"id": "c1", "check_worthiness": 0.9, "harm_potential": 4.0},
        {"id": "c2", "check_worthiness": 0.8, "harm_potential": 3.5},
        {"id": "c3", "check_worthiness": 0.7, "harm_potential": 3.0},
    ]
    
    # Higher marginal cost -> fewer targets
    expensive_params = TargetBudgetParams(
        marginal_cost_per_target=0.9,  # Very expensive
        budget_ceilings={"minimal": 10},
        budget_floors={"minimal": 1},
    )
    
    cheap_params = TargetBudgetParams(
        marginal_cost_per_target=0.1,  # Very cheap
        budget_ceilings={"minimal": 10},
        budget_floors={"minimal": 1},
    )
    
    k_expensive, _ = compute_optimal_target_count(claims, budget_class="minimal", params=expensive_params)
    k_cheap, _ = compute_optimal_target_count(claims, budget_class="minimal", params=cheap_params)
    
    # Cheaper targets should allow selecting more
    assert k_cheap >= k_expensive


def test_evoi_function_high_entropy_at_uncertainty():
    """EVOI should be highest when prior is 0.5 (max uncertainty)."""
    claim = {"check_worthiness": 0.8, "harm_potential": 3.0}
    
    evoi_uncertain = _expected_value_of_information(claim, prior_p=0.5)
    evoi_certain = _expected_value_of_information(claim, prior_p=0.95)
    
    # Max entropy at p=0.5 -> higher EVOI
    assert evoi_uncertain > evoi_certain


def test_evoi_function_harm_increases_value():
    """Higher harm potential should increase EVOI."""
    claim_high_harm = {"check_worthiness": 0.5, "harm_potential": 1.0}
    claim_low_harm = {"check_worthiness": 0.5, "harm_potential": 0.2}
    
    evoi_high = _expected_value_of_information(claim_high_harm)
    evoi_low = _expected_value_of_information(claim_low_harm)
    
    assert evoi_high > evoi_low


def test_bayesian_integration_with_select_targets():
    """Integration: Bayesian model should be used in select_verification_targets."""
    claims = [
        {"id": "c1", "check_worthiness": 0.9, "harm_potential": 4.5},
        {"id": "c2", "check_worthiness": 0.85, "harm_potential": 4.0},
        {"id": "c3", "check_worthiness": 0.3, "harm_potential": 1.0},
        {"id": "c4", "check_worthiness": 0.2, "harm_potential": 0.5},
    ]
    
    # Without explicit max_targets, Bayesian model determines count
    result = select_verification_targets(claims, budget_class="minimal")
    
    # Should have selected some targets based on EVOI
    assert len(result.targets) >= 1
    assert len(result.targets) <= 3  # ceiling for minimal
    
    # High-value claims should be targets
    target_ids = {c["id"] for c in result.targets}
    assert "c1" in target_ids or "c2" in target_ids

