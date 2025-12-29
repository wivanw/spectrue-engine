# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Retrieval quality evaluation helpers."""

from __future__ import annotations

from typing import Any

from spectrue_core.runtime_config import EngineRuntimeConfig
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.calibration_models import linear_score, logistic_score
from spectrue_core.verification.calibration_registry import CalibrationRegistry
from spectrue_core.verification.evidence_pack import score_evidence_likeness
from spectrue_core.verification.source_utils import extract_domain, score_source_quality


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_retrieval_state(
    sources: list[dict],
    *,
    calibration_registry: CalibrationRegistry | None = None,
) -> dict[str, float]:
    """
    Build retrieval state features without hard-coded weights.
    """
    relevance_scores: list[float] = []
    quote_count = 0
    temporal_flags: list[str] = []
    stance_set: set[str] = set()
    domains: set[str] = set()

    for src in sources:
        if not isinstance(src, dict):
            continue
        score = src.get("relevance_score")
        if score is not None:
            relevance_scores.append(_safe_float(score))
        if src.get("quote"):
            quote_count += 1
        stance = str(src.get("stance") or "").upper()
        if stance:
            stance_set.add(stance)
        temporal_flag = str(src.get("temporal_flag") or "")
        if temporal_flag:
            temporal_flags.append(temporal_flag.lower())
        url = str(src.get("url") or src.get("link") or "")
        domain = extract_domain(url) or str(src.get("domain") or "")
        if domain:
            domains.add(domain)

    relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    evidence_likeness = score_evidence_likeness(
        sources,
        calibration_registry=calibration_registry,
    )
    source_quality = score_source_quality(sources)
    total_sources = max(1, len([s for s in sources if isinstance(s, dict)]))
    domain_diversity = _clamp01(len(domains) / total_sources) if domains else 0.0
    quote_coverage = _clamp01(quote_count / total_sources) if total_sources else 0.0
    conflict_presence = 1.0 if ("SUPPORT" in stance_set and "REFUTE" in stance_set) else 0.0
    temporal_risk = 1.0 if any(flag in {"outdated", "future"} for flag in temporal_flags) else 0.0

    return {
        "mean_relevance": _clamp01(relevance),
        "evidence_likeness": _clamp01(evidence_likeness),
        "source_quality": _clamp01(source_quality),
        "domain_diversity": domain_diversity,
        "conflict_presence": conflict_presence,
        "quote_coverage": quote_coverage,
        "temporal_risk": temporal_risk,
    }


def predict_marginal_gain(
    state: dict[str, float],
    *,
    calibration_registry: CalibrationRegistry | None = None,
) -> dict[str, Any]:
    registry = calibration_registry or CalibrationRegistry.from_runtime(None)
    model = registry.get_model("retrieval_gain")
    features = {
        "relevance_gap": 1.0 - _clamp01(state.get("mean_relevance", 0.0)),
        "evidence_gap": 1.0 - _clamp01(state.get("evidence_likeness", 0.0)),
        "quality_gap": 1.0 - _clamp01(state.get("source_quality", 0.0)),
        "diversity_gap": 1.0 - _clamp01(state.get("domain_diversity", 0.0)),
        "quote_gap": 1.0 - _clamp01(state.get("quote_coverage", 0.0)),
        "conflict_presence": _clamp01(state.get("conflict_presence", 0.0)),
        "temporal_risk": _clamp01(state.get("temporal_risk", 0.0)),
    }
    if not model:
        policy = registry.policy.retrieval_gain
        raw, expected_gain = logistic_score(
            features,
            policy.fallback_weights or policy.weights,
            bias=policy.fallback_bias or policy.bias,
        )
        trace = {
            "model": "retrieval_gain",
            "version": policy.version,
            "features": features,
            "weights": policy.fallback_weights or policy.weights,
            "bias": policy.fallback_bias or policy.bias,
            "raw_score": raw,
            "score": expected_gain,
            "fallback_used": True,
        }
        return {
            "expected_gain": _clamp01(expected_gain),
            "features": features,
            "trace": trace,
        }
    expected_gain, trace = model.score(features)
    return {
        "expected_gain": _clamp01(expected_gain),
        "features": features,
        "trace": trace,
    }


def evaluate_retrieval_confidence(
    sources: list[dict],
    *,
    runtime_config: EngineRuntimeConfig | None = None,
    expected_cost: float | None = None,
) -> dict[str, Any]:
    """
    Compute retrieval confidence and expected marginal gain from state features.
    """
    registry = CalibrationRegistry.from_runtime(runtime_config)
    state = build_retrieval_state(sources, calibration_registry=registry)

    model = registry.get_model("retrieval_confidence")
    confidence_features = {
        "mean_relevance": state["mean_relevance"],
        "evidence_likeness": state["evidence_likeness"],
        "source_quality": state["source_quality"],
    }
    if model:
        retrieval_confidence, confidence_trace = model.score(confidence_features)
    else:
        policy = registry.policy.retrieval_confidence
        raw, retrieval_confidence = linear_score(
            confidence_features,
            policy.fallback_weights or policy.weights,
            bias=policy.fallback_bias or policy.bias,
            clamp=True,
        )
        confidence_trace = {
            "model": "retrieval_confidence",
            "version": policy.version,
            "features": confidence_features,
            "weights": policy.fallback_weights or policy.weights,
            "bias": policy.fallback_bias or policy.bias,
            "raw_score": raw,
            "score": retrieval_confidence,
            "fallback_used": True,
        }

    gain_payload = predict_marginal_gain(state, calibration_registry=registry)
    expected_gain = gain_payload["expected_gain"]
    expected_cost_raw = _safe_float(expected_cost, default=registry.policy.retrieval_cost_norm)
    cost_norm = max(1e-6, float(registry.policy.retrieval_cost_norm))
    expected_cost_norm = _safe_float(expected_cost_raw / cost_norm, default=1.0)
    cost_weight = max(1e-6, float(registry.policy.retrieval_cost_weight))
    value_per_cost = expected_gain / max(expected_cost_norm * cost_weight, 1e-6)

    result = {
        "relevance_score": state["mean_relevance"],
        "evidence_likeness_score": state["evidence_likeness"],
        "source_quality_score": state["source_quality"],
        "domain_diversity": state["domain_diversity"],
        "conflict_presence": state["conflict_presence"],
        "quote_coverage": state["quote_coverage"],
        "temporal_risk": state["temporal_risk"],
        "retrieval_confidence": _clamp01(retrieval_confidence),
        "retrieval_confidence_trace": confidence_trace,
        "expected_gain": _clamp01(expected_gain),
        "expected_gain_trace": gain_payload.get("trace"),
        "expected_cost": _clamp01(expected_cost_norm),
        "expected_cost_raw": expected_cost_raw,
        "value_per_cost": float(value_per_cost),
    }

    Trace.event(
        "retrieval.state",
        {
            "features": state,
            "expected_gain": expected_gain,
            "expected_cost": expected_cost_norm,
            "value_per_cost": value_per_cost,
        },
    )
    return result
