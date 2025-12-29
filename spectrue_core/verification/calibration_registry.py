# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Calibration registry for learned scoring models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectrue_core.runtime_config import CalibrationPolicyConfig, CalibrationModelPolicy, EngineRuntimeConfig
from spectrue_core.verification.calibration_models import (
    build_trace_payload,
    linear_score,
    logistic_score,
)


@dataclass(frozen=True)
class CalibrationModel:
    name: str
    policy: CalibrationModelPolicy

    def score(self, features: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        use_fallback = (not self.policy.enabled) or (not self.policy.weights)
        weights = self.policy.fallback_weights if use_fallback else self.policy.weights
        bias = self.policy.fallback_bias if use_fallback else self.policy.bias
        mode = (self.policy.score_mode or "sigmoid").lower()

        if mode == "linear":
            raw_score, score = linear_score(features, weights, bias=bias, clamp=True)
        else:
            raw_score, score = logistic_score(features, weights, bias=bias)

        trace = build_trace_payload(
            model_name=self.name,
            model_version=self.policy.version,
            features=features,
            weights=weights,
            bias=bias,
            raw_score=raw_score,
            score=score,
            fallback_used=use_fallback,
            extra={"score_mode": mode},
        )
        return score, trace


@dataclass(frozen=True)
class CalibrationRegistry:
    policy: CalibrationPolicyConfig
    models: dict[str, CalibrationModel] = field(default_factory=dict)

    def get_model(self, name: str) -> CalibrationModel | None:
        return self.models.get(name)

    @classmethod
    def from_runtime(cls, runtime: EngineRuntimeConfig | None) -> "CalibrationRegistry":
        policy = getattr(runtime, "calibration", None) or CalibrationPolicyConfig()
        models = {
            "claim_utility": CalibrationModel("claim_utility", policy.claim_utility),
            "retrieval_confidence": CalibrationModel("retrieval_confidence", policy.retrieval_confidence),
            "retrieval_gain": CalibrationModel("retrieval_gain", policy.retrieval_gain),
            "evidence_likeness": CalibrationModel("evidence_likeness", policy.evidence_likeness),
            "search_relevance": CalibrationModel("search_relevance", policy.search_relevance),
        }
        return cls(policy=policy, models=models)
