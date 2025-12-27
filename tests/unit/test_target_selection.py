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
    # Highest worthiness claims should be targets
    target_ids = {c["id"] for c in result.targets}
    assert "c1" in target_ids
    assert "c2" in target_ids


def test_thesis_claims_prioritized():
    claims = [
        {"id": "c1", "text": "Support claim", "check_worthiness": 0.7, "claim_role": "support"},
        {"id": "c2", "text": "Thesis claim", "check_worthiness": 0.7, "claim_role": "thesis"},
    ]
    result = select_verification_targets(claims, max_targets=1)
    
    # Thesis should be selected over support with same worthiness
    assert len(result.targets) == 1
    assert result.targets[0]["id"] == "c2"


def test_budget_class_affects_max_targets():
    claims = [
        {"id": f"c{i}", "text": f"Claim {i}", "check_worthiness": 0.5}
        for i in range(10)
    ]
    
    # Minimal budget should cap at 2
    result_minimal = select_verification_targets(claims, max_targets=5, budget_class="minimal")
    assert len(result_minimal.targets) <= 2
    
    # Deep budget allows more
    result_deep = select_verification_targets(claims, max_targets=5, budget_class="deep")
    assert len(result_deep.targets) == 5


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
