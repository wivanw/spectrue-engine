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
Unit tests for Pipeline Mode and Invariant Steps.

Tests:
- PipelineMode frozen dataclass behavior
- Mode registry and get_mode()
- AssertSingleClaimStep invariant
- AssertSingleLanguageStep invariant
- PipelineViolation exception structure
"""

import pytest
from spectrue_core.verification.search import SearchDepth
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.pipeline import (
    GENERAL_MODE,
    DEEP_MODE,
    get_mode,
    PipelineContext,
    PipelineViolation,
)
from spectrue_core.pipeline.steps import (
    AssertSingleClaimStep,
    AssertSingleLanguageStep,
    AssertNonEmptyClaimsStep,
)


class TestPipelineMode:
    """Tests for PipelineMode dataclass."""

    def test_general_mode_is_frozen(self):
        """GENERAL_MODE should be immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            GENERAL_MODE.allow_batch = True

    def test_general_mode_properties(self):
        """General mode should have expected invariant values."""
        assert GENERAL_MODE.name == AnalysisMode.GENERAL
        assert GENERAL_MODE.allow_batch is False
        assert GENERAL_MODE.allow_clustering is False
        assert GENERAL_MODE.require_single_language is True
        assert GENERAL_MODE.max_claims_for_scoring == 1
        assert GENERAL_MODE.search_depth == SearchDepth.BASIC.value

    def test_deep_mode_properties(self):
        """Deep mode should have expected invariant values."""
        assert DEEP_MODE.name == AnalysisMode.DEEP
        assert DEEP_MODE.allow_batch is True
        assert DEEP_MODE.allow_clustering is True
        assert DEEP_MODE.require_single_language is False
        assert DEEP_MODE.max_claims_for_scoring == 0  # unlimited
        assert DEEP_MODE.search_depth == SearchDepth.ADVANCED.value

    def test_get_mode_general(self):
        """get_mode should return GENERAL_MODE for 'general'."""
        mode = get_mode(AnalysisMode.GENERAL)
        assert mode is GENERAL_MODE

    def test_get_mode_deep(self):
        """get_mode should return DEEP_MODE for 'deep'."""
        mode = get_mode(AnalysisMode.DEEP)
        assert mode is DEEP_MODE

    def test_get_mode_invalid_raises(self):
        """get_mode should raise ValueError for unknown mode."""
        with pytest.raises(ValueError, match="Unknown pipeline mode"):
            get_mode("invalid_mode")

    def test_mode_str_representation(self):
        """Mode should have readable str representation."""
        assert str(GENERAL_MODE) == "PipelineMode(general)"
        assert str(DEEP_MODE) == "PipelineMode(deep)"


class TestAssertSingleClaimStep:
    """Tests for AssertSingleClaimStep invariant."""

    @pytest.mark.asyncio
    async def test_passes_with_single_claim(self):
        """Step should pass when exactly one claim is present."""
        ctx = PipelineContext(
            mode=GENERAL_MODE,
            claims=[{"id": "c1", "text": "Test claim"}],
        )
        step = AssertSingleClaimStep()
        result = await step.run(ctx)
        assert result.claims == ctx.claims  # Context unchanged

    @pytest.mark.asyncio
    async def test_fails_with_no_claims(self):
        """Step should fail when no claims are present."""
        ctx = PipelineContext(mode=GENERAL_MODE, claims=[])
        step = AssertSingleClaimStep()
        with pytest.raises(PipelineViolation) as exc_info:
            await step.run(ctx)
        assert exc_info.value.expected == 1
        assert exc_info.value.actual == 0
        assert "single_claim" in exc_info.value.invariant

    @pytest.mark.asyncio
    async def test_fails_with_multiple_claims(self):
        """Step should fail when multiple claims are present."""
        ctx = PipelineContext(
            mode=GENERAL_MODE,
            claims=[
                {"id": "c1", "text": "Claim 1"},
                {"id": "c2", "text": "Claim 2"},
                {"id": "c3", "text": "Claim 3"},
            ],
        )
        step = AssertSingleClaimStep()
        with pytest.raises(PipelineViolation) as exc_info:
            await step.run(ctx)
        assert exc_info.value.expected == 1
        assert exc_info.value.actual == 3


class TestAssertSingleLanguageStep:
    """Tests for AssertSingleLanguageStep invariant."""

    @pytest.mark.asyncio
    async def test_passes_with_single_language(self):
        """Step should pass when all claims have same language."""
        ctx = PipelineContext(
            mode=GENERAL_MODE,
            claims=[{"id": "c1", "lang": "en", "text": "Test"}],
            lang="en",
        )
        step = AssertSingleLanguageStep()
        result = await step.run(ctx)
        assert result.claims == ctx.claims

    @pytest.mark.asyncio
    async def test_passes_with_no_claim_lang_uses_context_lang(self):
        """Step should use context lang when claims don't specify."""
        ctx = PipelineContext(
            mode=GENERAL_MODE,
            claims=[{"id": "c1", "text": "Test"}],  # No lang field
            lang="uk",
        )
        step = AssertSingleLanguageStep()
        result = await step.run(ctx)
        assert result.lang == "uk"

    @pytest.mark.asyncio
    async def test_fails_with_multiple_languages(self):
        """Step should fail when claims have different languages."""
        ctx = PipelineContext(
            mode=GENERAL_MODE,
            claims=[
                {"id": "c1", "lang": "en", "text": "English claim"},
                {"id": "c2", "lang": "uk", "text": "Українська заява"},
            ],
        )
        step = AssertSingleLanguageStep()
        with pytest.raises(PipelineViolation) as exc_info:
            await step.run(ctx)
        assert "single_language" in exc_info.value.invariant
        assert "en" in str(exc_info.value.actual)
        assert "uk" in str(exc_info.value.actual)


class TestAssertNonEmptyClaimsStep:
    """Tests for AssertNonEmptyClaimsStep invariant."""

    @pytest.mark.asyncio
    async def test_passes_with_claims(self):
        """Step should pass when at least one claim is present."""
        ctx = PipelineContext(
            mode=DEEP_MODE,
            claims=[{"id": "c1", "text": "Test"}],
        )
        step = AssertNonEmptyClaimsStep()
        result = await step.run(ctx)
        assert len(result.claims) == 1

    @pytest.mark.asyncio
    async def test_fails_with_empty_claims(self):
        """Step should fail when no claims are present."""
        ctx = PipelineContext(mode=DEEP_MODE, claims=[])
        step = AssertNonEmptyClaimsStep()
        with pytest.raises(PipelineViolation) as exc_info:
            await step.run(ctx)
        assert "claims_required" in exc_info.value.invariant
        assert exc_info.value.actual == 0


class TestPipelineViolation:
    """Tests for PipelineViolation exception."""

    def test_exception_message_format(self):
        """Exception should have readable message."""
        exc = PipelineViolation(
            step_name="test_step",
            invariant="test_invariant",
            expected="foo",
            actual="bar",
        )
        assert "test_step" in str(exc)
        assert "test_invariant" in str(exc)
        assert "foo" in str(exc)
        assert "bar" in str(exc)

    def test_to_trace_dict(self):
        """Exception should serialize to trace dict."""
        exc = PipelineViolation(
            step_name="test_step",
            invariant="test_invariant",
            expected=1,
            actual=3,
            details={"mode": "general"},
        )
        trace_dict = exc.to_trace_dict()
        assert trace_dict["error"] == "pipeline_violation"
        assert trace_dict["step_name"] == "test_step"
        assert trace_dict["invariant"] == "test_invariant"
        assert trace_dict["details"]["mode"] == "general"


class TestPipelineContext:
    """Tests for PipelineContext immutability."""

    def test_with_update_returns_new_context(self):
        """with_update should return new context, not mutate."""
        ctx = PipelineContext(mode=GENERAL_MODE, claims=[{"id": "c1"}])
        new_ctx = ctx.with_update(claims=[{"id": "c2"}])

        assert ctx.claims == [{"id": "c1"}]  # Original unchanged
        assert new_ctx.claims == [{"id": "c2"}]  # New has update

    def test_set_extra_returns_new_context(self):
        """set_extra should return new context with extra field."""
        ctx = PipelineContext(mode=GENERAL_MODE)
        new_ctx = ctx.set_extra("foo", "bar")

        assert ctx.get_extra("foo") is None  # Original unchanged
        assert new_ctx.get_extra("foo") == "bar"

    def test_get_extra_default(self):
        """get_extra should return default when key missing."""
        ctx = PipelineContext(mode=GENERAL_MODE)
        assert ctx.get_extra("missing", "default") == "default"
