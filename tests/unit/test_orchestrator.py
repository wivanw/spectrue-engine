# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Unit Tests for ClaimOrchestrator

Tests phase generation logic based on claim metadata.
"""

import pytest

from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
from spectrue_core.verification.orchestration.execution_plan import BudgetClass
from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    ClaimRole,
    VerificationTarget,
    MetadataConfidence,
    SearchLocalePlan,
    RetrievalPolicy,
    EvidenceChannel,
)
from spectrue_core.verification.evidence.evidence_pack import Claim


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def orchestrator():
    """Create a ClaimOrchestrator instance."""
    return ClaimOrchestrator()


def make_claim(
    claim_id: str,
    verification_target: VerificationTarget = VerificationTarget.REALITY,
    claim_role: ClaimRole = ClaimRole.CORE,
    metadata_confidence: MetadataConfidence = MetadataConfidence.HIGH,
    primary_locale: str = "en",
    fallback_locales: list[str] | None = None,
) -> Claim:
    """Helper to create a claim with metadata."""
    metadata = ClaimMetadata(
        verification_target=verification_target,
        claim_role=claim_role,
        check_worthiness=0.8,
        search_locale_plan=SearchLocalePlan(
            primary=primary_locale,
            fallback=fallback_locales or ["en"],
        ),
        retrieval_policy=RetrievalPolicy(
            channels_allowed=[EvidenceChannel.AUTHORITATIVE, EvidenceChannel.REPUTABLE_NEWS],
        ),
        metadata_confidence=metadata_confidence,
    )
    
    return Claim(
        id=claim_id,
        text="Test claim text",
        normalized_text="Test claim normalized",
        type="core",
        topic_group="Science",
        topic_key="Test Topic",
        importance=0.8,
        check_worthiness=0.8,
        evidence_requirement={
            "needs_primary_source": False,
            "needs_independent_2x": True,
        },
        search_queries=["test query"],
        query_candidates=[],
        metadata=metadata,
    )


# ─────────────────────────────────────────────────────────────────────────────
# T13: Reality Claims Phase Generation
# ─────────────────────────────────────────────────────────────────────────────

class TestRealityClaims:
    """Test phase generation for reality-checking claims."""
    
    def test_reality_claim_deep_budget_gets_4_phases(self, orchestrator):
        """T13: Reality claim with DEEP budget → 4 phases (A/B/C/D)."""
        claims = [make_claim(
            "c1",
            verification_target=VerificationTarget.REALITY,
            primary_locale="uk",
            fallback_locales=["en"],
        )]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.DEEP)
        
        phases = plan.get_phases("c1")
        phase_ids = [p.phase_id for p in phases]
        
        # Should have A, B, C (fallback), D
        assert len(phases) == 4, f"Expected 4 phases, got {len(phases)}: {phase_ids}"
        assert phase_ids == ["A", "B", "C", "D"], f"Expected [A, B, C, D], got {phase_ids}"
    
    def test_reality_claim_standard_budget_gets_2_phases(self, orchestrator):
        """Reality claim with STANDARD budget → 2 phases (A/B)."""
        claims = [make_claim("c1", verification_target=VerificationTarget.REALITY)]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
        
        phases = plan.get_phases("c1")
        phase_ids = [p.phase_id for p in phases]
        
        assert len(phases) == 2, f"Expected 2 phases, got {len(phases)}: {phase_ids}"
        assert phase_ids == ["A", "B"], f"Expected [A, B], got {phase_ids}"
    
    def test_reality_claim_minimal_budget_gets_1_phase(self, orchestrator):
        """T13: Reality claim with MINIMAL budget → 1 phase (A only)."""
        claims = [make_claim("c1", verification_target=VerificationTarget.REALITY)]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.MINIMAL)
        
        phases = plan.get_phases("c1")
        phase_ids = [p.phase_id for p in phases]
        
        assert len(phases) == 1, f"Expected 1 phase, got {len(phases)}: {phase_ids}"
        assert phase_ids == ["A"], f"Expected [A], got {phase_ids}"


# ─────────────────────────────────────────────────────────────────────────────
# T13: None Target Claims
# ─────────────────────────────────────────────────────────────────────────────

class TestNoneClaims:
    """Test phase generation for non-verifiable claims."""
    
    def test_none_target_gets_no_phases(self, orchestrator):
        """T13: Claim with verification_target=NONE → 0 phases (no search)."""
        claims = [make_claim(
            "c1",
            verification_target=VerificationTarget.NONE,
            claim_role=ClaimRole.CONTEXT,
        )]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.DEEP)
        
        phases = plan.get_phases("c1")
        
        assert len(phases) == 0, f"Expected 0 phases for none target, got {len(phases)}"
    
    def test_none_target_low_confidence_gets_a_light(self, orchestrator):
        """T13: None target + LOW confidence → 1 phase (A-light) for fail-open."""
        claims = [make_claim(
            "c1",
            verification_target=VerificationTarget.NONE,
            metadata_confidence=MetadataConfidence.LOW,
        )]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.DEEP)
        
        phases = plan.get_phases("c1")
        phase_ids = [p.phase_id for p in phases]
        
        assert len(phases) == 1, f"Expected 1 phase for fail-open, got {len(phases)}"
        assert phase_ids == ["A-light"], f"Expected [A-light], got {phase_ids}"


# ─────────────────────────────────────────────────────────────────────────────
# T13: Fail-Open (Low Confidence)
# ─────────────────────────────────────────────────────────────────────────────

class TestFailOpen:
    """Test fail-open behavior when metadata_confidence is LOW."""
    
    def test_low_confidence_prepends_a_light(self, orchestrator):
        """T13: Low confidence reality claim → A-light injected at start."""
        claims = [make_claim(
            "c1",
            verification_target=VerificationTarget.REALITY,
            metadata_confidence=MetadataConfidence.LOW,
        )]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
        
        phases = plan.get_phases("c1")
        phase_ids = [p.phase_id for p in phases]
        
        # Should have A-light prepended: [A-light, A, B]
        assert phases[0].phase_id == "A-light", f"First phase should be A-light, got {phase_ids}"
        assert "A" in phase_ids, "Should include Phase A"
        assert "B" in phase_ids, "Should include Phase B"
    
    def test_high_confidence_no_a_light(self, orchestrator):
        """High confidence → no A-light injection."""
        claims = [make_claim(
            "c1",
            verification_target=VerificationTarget.REALITY,
            metadata_confidence=MetadataConfidence.HIGH,
        )]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
        
        phases = plan.get_phases("c1")
        phase_ids = [p.phase_id for p in phases]
        
        assert "A-light" not in phase_ids, "A-light should not be present for high confidence"


# ─────────────────────────────────────────────────────────────────────────────
# T13: Attribution Claims
# ─────────────────────────────────────────────────────────────────────────────

class TestAttributionClaims:
    """Test phase generation for attribution claims."""
    
    def test_attribution_gets_origin_phase(self, orchestrator):
        """Attribution claim → uses A-origin phase."""
        claims = [make_claim(
            "c1",
            verification_target=VerificationTarget.ATTRIBUTION,
        )]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
        
        phases = plan.get_phases("c1")
        phase_ids = [p.phase_id for p in phases]
        
        assert "A-origin" in phase_ids, f"Attribution should use A-origin phase: {phase_ids}"


# ─────────────────────────────────────────────────────────────────────────────
# T13: Multiple Claims
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleClaims:
    """Test ExecutionPlan with multiple claims."""
    
    def test_mixed_claims_get_different_phases(self, orchestrator):
        """Mixed claims (reality + none) get appropriate phases."""
        claims = [
            make_claim("c1", verification_target=VerificationTarget.REALITY),
            make_claim("c2", verification_target=VerificationTarget.NONE, claim_role=ClaimRole.CONTEXT),
            make_claim("c3", verification_target=VerificationTarget.ATTRIBUTION),
        ]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
        
        # c1 (reality): should have A, B
        c1_phases = [p.phase_id for p in plan.get_phases("c1")]
        assert "A" in c1_phases, f"Reality claim should have Phase A: {c1_phases}"
        
        # c2 (none): should have 0 phases
        c2_phases = plan.get_phases("c2")
        assert len(c2_phases) == 0, f"None claim should have 0 phases: {c2_phases}"
        
        # c3 (attribution): should have A-origin
        c3_phases = [p.phase_id for p in plan.get_phases("c3")]
        assert "A-origin" in c3_phases, f"Attribution claim should have A-origin: {c3_phases}"
    
    def test_plan_summary(self, orchestrator):
        """ExecutionPlan.summary() returns readable string."""
        claims = [
            make_claim("c1", verification_target=VerificationTarget.REALITY),
            make_claim("c2", verification_target=VerificationTarget.NONE),
        ]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.MINIMAL)
        
        summary = plan.summary()
        assert "c1" in summary
        assert "ExecutionPlan" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_claims_list(self, orchestrator):
        """Empty claims list → empty plan."""
        plan = orchestrator.build_execution_plan([], BudgetClass.STANDARD)
        
        assert plan.total_phases == 0
        assert plan.max_depth == 0
    
    def test_claim_without_metadata(self, orchestrator):
        """Claim without metadata → treated as reality claim."""
        claims = [Claim(
            id="c1",
            text="Test claim",
            normalized_text="Test claim",
            type="core",
            topic_group="Science",
            topic_key="Test",
            importance=0.8,
            check_worthiness=0.8,
            evidence_requirement={},
            search_queries=["test"],
            query_candidates=[],
            # metadata=None (not set)
        )]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
        
        phases = plan.get_phases("c1")
        assert len(phases) > 0, "Claim without metadata should still get phases"
    
    def test_locale_same_as_fallback(self, orchestrator):
        """When primary=fallback, Phase C is skipped."""
        claims = [make_claim(
            "c1",
            verification_target=VerificationTarget.REALITY,
            primary_locale="en",
            fallback_locales=["en"],  # Same as primary
        )]
        
        plan = orchestrator.build_execution_plan(claims, BudgetClass.DEEP)
        
        phases = plan.get_phases("c1")
        phase_ids = [p.phase_id for p in phases]
        
        # Should NOT have Phase C since locale == fallback
        assert "C" not in phase_ids, f"Should skip Phase C when locale equals fallback: {phase_ids}"
