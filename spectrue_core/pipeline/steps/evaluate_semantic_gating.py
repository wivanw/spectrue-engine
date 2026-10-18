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

"""Semantic gating step: verifiability gate G2 (M145 Phase R).

Runs before target selection. Scales each claim's `check_worthiness` by the
calibrated P(checkable), so the Bayesian target-selection EVOI decides which
claims get searched. Logic: `use_cases.retrieval.decision_gates.apply_verifiability`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.runtime_config import DecisionSensorCalibration, DecisionsConfig, uncalibrated_sensors
from spectrue_core.use_cases.retrieval.decision_gates import apply_verifiability


@dataclass
class EvaluateSemanticGatingStep:
    agent: Any
    decisions: DecisionsConfig = field(default_factory=DecisionsConfig)  # all gates off
    sensors: dict[str, DecisionSensorCalibration] = field(default_factory=uncalibrated_sensors)
    name: str = "semantic_gating"
    weight: float = 1.0

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        claims = ctx.get_extra("eligible_claims", ctx.claims or [])
        updated = await apply_verifiability(self.agent, self.decisions, self.sensors, list(claims), ctx.lang or "en")
        return ctx if updated is None else ctx.set_extra("eligible_claims", updated)
