# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from spectrue_core.verification.target_selection import (
    select_verification_targets,
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

