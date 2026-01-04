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
from spectrue_core.pipeline.contracts import (
    CLAIMS_KEY,
    EVIDENCE_INDEX_KEY,
    INPUT_DOC_KEY,
    JUDGMENTS_KEY,
    ClaimItem,
    Claims,
    EvidenceIndex,
    EvidenceItem,
    EvidencePackContract,
    InputDoc,
    Judgments,
)
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


def _assert_instance(value: object, expected: type, key: str) -> None:
    if not isinstance(value, expected):
        raise PipelineViolation(
            step_name="assert_contracts",
            invariant="contract_type_mismatch",
            expected=expected.__name__,
            actual=type(value).__name__,
            details={"key": key},
        )


def _assert_claims_contract(value: Claims) -> None:
    for item in value.claims:
        if not isinstance(item, ClaimItem):
            raise PipelineViolation(
                step_name="assert_contracts",
                invariant="contract_claim_item_type",
                expected="ClaimItem",
                actual=type(item).__name__,
                details={"claim_id": getattr(item, "id", None)},
            )


def _assert_evidence_pack(pack: EvidencePackContract) -> None:
    for item in pack.items:
        if not isinstance(item, EvidenceItem):
            raise PipelineViolation(
                step_name="assert_contracts",
                invariant="contract_evidence_item_type",
                expected="EvidenceItem",
                actual=type(item).__name__,
                details={"url": getattr(item, "url", None)},
            )


def _assert_evidence_index(value: EvidenceIndex) -> None:
    if value.global_pack is not None:
        _assert_evidence_pack(value.global_pack)
    for pack in value.by_claim_id.values():
        _assert_evidence_pack(pack)


@dataclass
class AssertContractPresenceStep:
    """
    Assert that required contract objects are present and correctly typed.

    This enforces stable step boundaries with no implicit dict payloads.
    """

    required: tuple[str, ...] = (
        INPUT_DOC_KEY,
        CLAIMS_KEY,
        EVIDENCE_INDEX_KEY,
        JUDGMENTS_KEY,
    )
    name: str = "assert_contracts"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        contract_map: dict[str, type] = {
            INPUT_DOC_KEY: InputDoc,
            CLAIMS_KEY: Claims,
            EVIDENCE_INDEX_KEY: EvidenceIndex,
            JUDGMENTS_KEY: Judgments,
        }

        for key in self.required:
            value = ctx.get_extra(key)
            if value is None:
                raise PipelineViolation(
                    step_name=self.name,
                    invariant="contract_missing",
                    expected=contract_map.get(key, object).__name__,
                    actual="missing",
                    details={"key": key},
                )

            expected = contract_map.get(key)
            if expected:
                _assert_instance(value, expected, key)

            if key == CLAIMS_KEY:
                _assert_claims_contract(value)
            elif key == EVIDENCE_INDEX_KEY:
                _assert_evidence_index(value)
            elif key == JUDGMENTS_KEY and isinstance(value, Judgments):
                if value.standard is not None and not isinstance(value.standard, dict):
                    raise PipelineViolation(
                        step_name=self.name,
                        invariant="contract_judgments_standard_type",
                        expected="dict",
                        actual=type(value.standard).__name__,
                        details={"key": key},
                    )
                for item in value.deep:
                    if not isinstance(item, dict):
                        raise PipelineViolation(
                            step_name=self.name,
                            invariant="contract_judgments_deep_type",
                            expected="dict",
                            actual=type(item).__name__,
                            details={"key": key},
                        )

        return ctx


@dataclass
class AssertStandardResultKeysStep:
    """
    Assert that standard-mode responses include legacy UI fields.

    Ensures the standard payload shape is stable and excludes deep-only data.
    """

    required: tuple[str, ...] = ("text", "details", "anchor_claim", "rgba", "sources")
    name: str = "assert_standard_result_keys"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        final_result = ctx.get_extra("final_result")
        if not isinstance(final_result, dict):
            raise PipelineViolation(
                step_name=self.name,
                invariant="standard_result_missing",
                expected="dict",
                actual=type(final_result).__name__,
                details={"mode": ctx.mode.name},
            )

        missing = [key for key in self.required if key not in final_result]
        if missing:
            raise PipelineViolation(
                step_name=self.name,
                invariant="standard_result_keys_missing",
                expected=self.required,
                actual=tuple(missing),
                details={"mode": ctx.mode.name},
            )

        if "deep_analysis" in final_result:
            raise PipelineViolation(
                step_name=self.name,
                invariant="standard_result_contains_deep_payload",
                expected="deep_analysis absent",
                actual="deep_analysis present",
                details={"mode": ctx.mode.name},
            )

        if not (final_result.get("rationale") or final_result.get("analysis")):
            raise PipelineViolation(
                step_name=self.name,
                invariant="standard_result_missing_rationale",
                expected="rationale or analysis",
                actual="missing",
                details={"mode": ctx.mode.name},
            )

        if "cost_summary" not in final_result and "credits" not in final_result:
            raise PipelineViolation(
                step_name=self.name,
                invariant="standard_result_missing_cost_summary",
                expected="cost_summary or credits",
                actual="missing",
                details={"mode": ctx.mode.name},
            )

        rgba = final_result.get("rgba")
        if not (isinstance(rgba, list) and len(rgba) == 4):
            raise PipelineViolation(
                step_name=self.name,
                invariant="standard_result_invalid_rgba",
                expected="list[4]",
                actual=type(rgba).__name__,
                details={"mode": ctx.mode.name, "length": len(rgba) if isinstance(rgba, list) else None},
            )

        details = final_result.get("details")
        if not isinstance(details, list):
            raise PipelineViolation(
                step_name=self.name,
                invariant="standard_result_invalid_details",
                expected="list",
                actual=type(details).__name__,
                details={"mode": ctx.mode.name},
            )

        anchor = final_result.get("anchor_claim")
        if not isinstance(anchor, dict) or not anchor.get("id") or "text" not in anchor:
            raise PipelineViolation(
                step_name=self.name,
                invariant="standard_result_invalid_anchor",
                expected="anchor_claim with id/text",
                actual=anchor,
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
