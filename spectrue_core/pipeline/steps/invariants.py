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
Invariant Steps

Steps that validate pipeline invariants before processing.
If an invariant is violated, PipelineViolation is raised.

These steps act as gates — they ensure the input matches
the pipeline mode's requirements before expensive work begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineViolation


@dataclass
class AssertSingleClaimStep:
    """
    Assert that exactly one claim is present.

    This invariant is required for normal mode, where only
    a single primary claim should be verified.

    Raises:
        PipelineViolation: If claim count != 1
    """

    name: str = "assert_single_claim"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        claim_count = len(ctx.claims)

        if claim_count != 1:
            raise PipelineViolation(
                step_name=self.name,
                invariant="normal_pipeline_single_claim",
                expected=1,
                actual=claim_count,
                details={
                    "mode": ctx.mode.name,
                    "claim_ids": [c.get("id", "?") for c in ctx.claims[:5]],
                },
            )

        return ctx


@dataclass
class AssertMaxClaimsStep:
    """
    Assert that claim count does not exceed maximum.

    Configurable maximum for modes that allow batch but with limits.

    Raises:
        PipelineViolation: If claim count > max_claims
    """

    max_claims: int
    name: str = "assert_max_claims"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        claim_count = len(ctx.claims)

        if claim_count > self.max_claims:
            raise PipelineViolation(
                step_name=self.name,
                invariant="max_claims_exceeded",
                expected=f"<= {self.max_claims}",
                actual=claim_count,
                details={
                    "mode": ctx.mode.name,
                    "max_claims": self.max_claims,
                },
            )

        return ctx


@dataclass
class AssertSingleLanguageStep:
    """
    Assert that all claims share the same language.

    Normal mode requires single-language input to avoid
    cross-language search complexity.

    Raises:
        PipelineViolation: If multiple languages detected
    """

    name: str = "assert_single_language"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        # Collect languages from claims
        languages: set[str] = set()

        for claim in ctx.claims:
            claim_lang = claim.get("lang") or claim.get("language")
            if claim_lang:
                languages.add(claim_lang.lower()[:2])  # Normalize to 2-char

        # If claims don't have lang, use context lang
        if not languages:
            languages.add(ctx.lang.lower()[:2])

        if len(languages) > 1:
            raise PipelineViolation(
                step_name=self.name,
                invariant="single_language_required",
                expected="1 language",
                actual=f"{len(languages)} languages: {sorted(languages)}",
                details={
                    "mode": ctx.mode.name,
                    "languages": sorted(languages),
                },
            )

        return ctx


@dataclass
class AssertNonEmptyClaimsStep:
    """
    Assert that at least one claim is present.

    This is a baseline invariant for all pipelines — you cannot
    verify nothing.

    Raises:
        PipelineViolation: If no claims present
    """

    name: str = "assert_non_empty_claims"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.claims:
            raise PipelineViolation(
                step_name=self.name,
                invariant="claims_required",
                expected=">= 1",
                actual=0,
                details={"mode": ctx.mode.name},
            )

        return ctx


@dataclass
class AssertMeteringEnabledStep:
    """
    Assert that cost metering is available.

    Required for production pipelines to track usage.
    Checks that trace is present (metering hooks into trace).

    Raises:
        PipelineViolation: If trace/metering not available
    """

    name: str = "assert_metering_enabled"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.require_metering and ctx.trace is None:
            raise PipelineViolation(
                step_name=self.name,
                invariant="metering_required",
                expected="trace != None",
                actual="trace is None",
                details={"mode": ctx.mode.name},
            )

        return ctx


def get_invariant_steps_for_mode(mode_name: str) -> list[Any]:
    """
    Get the standard invariant steps for a mode.

    Args:
        mode_name: "normal" or "deep"

    Returns:
        List of invariant step instances
    """
    if mode_name == "normal":
        return [
            AssertNonEmptyClaimsStep(),
            AssertSingleClaimStep(),
            AssertSingleLanguageStep(),
        ]
    else:  # deep
        return [
            AssertNonEmptyClaimsStep(),
            # Deep mode allows batch and multi-language
        ]
