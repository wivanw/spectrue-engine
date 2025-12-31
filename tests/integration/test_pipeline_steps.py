# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Integration tests for Pipeline step-based execution.

Tests the full pipeline flow including:
- Invariant validation (single claim, single language)
- PipelineFactory step composition
- execute_pipeline entry point
"""

import pytest
from spectrue_core.pipeline import (
    validate_claims_for_mode,
    execute_pipeline,
    PipelineViolation,
    PipelineFactory,
    PipelineContext,
    NORMAL_MODE,
    DEEP_MODE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


class MockSearchMgr:
    """Minimal mock for SearchManager."""

    def calculate_cost(self, *_args, **_kwargs):
        return 0

    def get_search_meta(self):
        return {}


class MockAgent:
    """Minimal mock for FactCheckerAgent."""

    async def cluster_evidence(self, **_kwargs):
        return None

    async def score_evidence(self, pack, **_kwargs):
        return {"status": "ok", "verified_score": 0.5}


# ─────────────────────────────────────────────────────────────────────────────
# T014: Normal Pipeline Rejects Multi-Claim Input
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalPipelineInvariants:
    """Test that normal pipeline correctly validates invariants."""

    @pytest.mark.asyncio
    async def test_validate_normal_mode_single_claim_passes(self):
        """Normal mode validation should pass with exactly one claim."""
        claims = [{"id": "c1", "text": "Single claim", "lang": "en"}]

        # Should not raise
        await validate_claims_for_mode("normal", claims)

    @pytest.mark.asyncio
    async def test_validate_normal_mode_multi_claim_raises(self):
        """Normal mode validation should reject multiple claims."""
        claims = [
            {"id": "c1", "text": "First claim", "lang": "en"},
            {"id": "c2", "text": "Second claim", "lang": "en"},
        ]

        with pytest.raises(PipelineViolation) as exc_info:
            await validate_claims_for_mode("normal", claims)

        assert exc_info.value.step_name == "assert_single_claim"
        assert exc_info.value.expected == 1
        assert exc_info.value.actual == 2
        assert "single_claim" in exc_info.value.invariant

    @pytest.mark.asyncio
    async def test_validate_normal_mode_no_claims_raises(self):
        """Normal mode validation should reject empty claims."""
        claims = []

        with pytest.raises(PipelineViolation) as exc_info:
            await validate_claims_for_mode("normal", claims)

        assert "claims_required" in exc_info.value.invariant

    @pytest.mark.asyncio
    async def test_validate_normal_mode_multi_language_raises(self):
        """Normal mode validation should reject multi-language claims."""
        claims = [
            {"id": "c1", "text": "English claim", "lang": "en"},
            {"id": "c2", "text": "Українська заява", "lang": "uk"},
        ]

        with pytest.raises(PipelineViolation):
            await validate_claims_for_mode("normal", claims)

    @pytest.mark.asyncio
    async def test_validate_general_mode_alias_works(self):
        """'general' should be treated as alias for 'normal'."""
        claims = [{"id": "c1", "text": "Single claim"}]

        # Should not raise (general = normal)
        await validate_claims_for_mode("general", claims)


class TestDeepPipelineInvariants:
    """Test that deep pipeline allows batch claims."""

    @pytest.mark.asyncio
    async def test_validate_deep_mode_multi_claim_passes(self):
        """Deep mode validation should allow multiple claims."""
        claims = [
            {"id": "c1", "text": "First claim"},
            {"id": "c2", "text": "Second claim"},
            {"id": "c3", "text": "Third claim"},
        ]

        # Should not raise
        await validate_claims_for_mode("deep", claims)

    @pytest.mark.asyncio
    async def test_validate_deep_mode_no_claims_raises(self):
        """Deep mode still requires at least one claim."""
        claims = []

        with pytest.raises(PipelineViolation) as exc_info:
            await validate_claims_for_mode("deep", claims)

        assert "claims_required" in exc_info.value.invariant


class TestPipelineFactory:
    """Test PipelineFactory step composition."""

    def test_factory_builds_normal_pipeline_with_invariants(self):
        """Normal pipeline should include invariant steps."""
        factory = PipelineFactory(search_mgr=MockSearchMgr(), agent=MockAgent())
        pipeline = factory.build("normal")

        step_names = [s.name for s in pipeline.steps]

        # Should have invariant steps first
        assert "assert_non_empty_claims" in step_names
        assert "assert_single_claim" in step_names
        assert "assert_single_language" in step_names

        # Invariants should be before retrieval
        assert step_names.index("assert_single_claim") < step_names.index("legacy_phase_runner")

    def test_factory_builds_deep_pipeline_without_single_claim(self):
        """Deep pipeline should NOT have single-claim invariant."""
        factory = PipelineFactory(search_mgr=MockSearchMgr(), agent=MockAgent())
        pipeline = factory.build("deep")

        step_names = [s.name for s in pipeline.steps]

        # Should have non-empty check
        assert "assert_non_empty_claims" in step_names

        # Should NOT have single-claim check
        assert "assert_single_claim" not in step_names
        assert "assert_single_language" not in step_names

    def test_factory_from_profile_maps_correctly(self):
        """build_from_profile should map profile names to modes."""
        factory = PipelineFactory(search_mgr=MockSearchMgr(), agent=MockAgent())

        normal = factory.build_from_profile("normal")
        deep = factory.build_from_profile("deep")
        general = factory.build_from_profile("general")

        assert normal.mode.name == "normal"
        assert deep.mode.name == "deep"
        assert general.mode.name == "normal"  # general is alias


class TestPipelineExecution:
    """Test pipeline execution flow."""

    @pytest.mark.asyncio
    async def test_pipeline_run_propagates_violation(self):
        """Pipeline.run() should propagate PipelineViolation."""
        factory = PipelineFactory(search_mgr=MockSearchMgr(), agent=MockAgent())
        pipeline = factory.build("normal")

        # Multi-claim context should fail
        ctx = PipelineContext(
            mode=NORMAL_MODE,
            claims=[{"id": "c1"}, {"id": "c2"}],
        )

        with pytest.raises(PipelineViolation):
            await pipeline.run(ctx)

    @pytest.mark.asyncio
    async def test_pipeline_run_single_claim_passes_invariants(self):
        """Pipeline with single claim should pass invariant steps."""
        factory = PipelineFactory(search_mgr=MockSearchMgr(), agent=MockAgent())
        pipeline = factory.build("normal")

        ctx = PipelineContext(
            mode=NORMAL_MODE,
            claims=[{"id": "c1", "text": "Test claim"}],
            lang="en",
        )

        # Note: This will fail at LegacyPhaseRunnerStep because we're using mocks
        # The point is that invariant steps pass
        try:
            await pipeline.run(ctx)
        except Exception as e:
            # Should NOT be PipelineViolation
            assert not isinstance(e, PipelineViolation)
