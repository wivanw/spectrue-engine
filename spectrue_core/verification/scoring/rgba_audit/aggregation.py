# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""Deterministic aggregation for RGBA audit outputs."""

from __future__ import annotations

import re
from typing import Any

from spectrue_core.schema.rgba_audit import (
    ClaimAudit,
    EvidenceAudit,
    RGBAResult,
    RGBAMetric,
    RGBAStatus,
    SourceCluster,
)
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.scoring.rgba_audit.config import RGBAAuditConfig, default_rgba_audit_config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _source_tokens(source: dict[str, Any]) -> set[str]:
    title = str(source.get("title") or "")
    snippet = str(source.get("snippet") or "")
    text = f"{title} {snippet}".strip()
    tokens = _tokenize(text)
    if tokens:
        return tokens
    fallback = str(source.get("source_id") or source.get("url") or "")
    return _tokenize(fallback) if fallback else set()


def cluster_sources(
    sources: list[dict[str, Any]],
    config: RGBAAuditConfig,
) -> list[SourceCluster]:
    clusters: list[dict[str, Any]] = []
    threshold = config.redundancy_jaccard_threshold

    for source in sources:
        source_id = str(source.get("source_id") or source.get("url") or "")
        if not source_id:
            source_id = f"source_{len(clusters) + 1}"

        tokens = _source_tokens(source)
        assigned = False

        for cluster in clusters:
            rep_tokens = cluster["tokens"]
            if _jaccard(tokens, rep_tokens) >= threshold:
                cluster["source_ids"].append(source_id)
                assigned = True
                break

        if not assigned:
            clusters.append(
                {
                    "cluster_id": source_id,
                    "representative_source_id": source_id,
                    "source_ids": [source_id],
                    "tokens": tokens,
                }
            )

    return [
        SourceCluster(
            cluster_id=cluster["cluster_id"],
            representative_source_id=cluster["representative_source_id"],
            source_ids=cluster["source_ids"],
            size=len(cluster["source_ids"]),
        )
        for cluster in clusters
    ]


def _source_reliability(source: dict[str, Any] | None, config: RGBAAuditConfig) -> float:
    if not source:
        return config.source_reliability_priors["unknown"]
    tier = str(source.get("tier") or "unknown").upper()
    return config.source_reliability_priors.get(tier, config.source_reliability_priors["unknown"])


def _coerce_claim_audits(raw: list[Any]) -> list[ClaimAudit]:
    audits: list[ClaimAudit] = []
    for entry in raw:
        if isinstance(entry, ClaimAudit):
            audits.append(entry)
        elif isinstance(entry, dict):
            audits.append(ClaimAudit(**entry))
    return audits


def _coerce_evidence_audits(raw: list[Any]) -> list[EvidenceAudit]:
    audits: list[EvidenceAudit] = []
    for entry in raw:
        if isinstance(entry, EvidenceAudit):
            audits.append(entry)
        elif isinstance(entry, dict):
            audits.append(EvidenceAudit(**entry))
    return audits


def _config_snapshot(config: RGBAAuditConfig) -> dict[str, Any]:
    return {
        "redundancy_jaccard_threshold": config.redundancy_jaccard_threshold,
        "conflict_threshold": config.conflict_threshold,
        "c0": config.c0,
        "c1": config.c1,
        "stance_weights": dict(config.stance_weights),
        "directness_weights": dict(config.directness_weights),
        "specificity_weights": dict(config.specificity_weights),
        "quote_integrity_weights": dict(config.quote_integrity_weights),
        "novelty_weights": dict(config.novelty_weights),
        "source_reliability_priors": dict(config.source_reliability_priors),
        "assertion_strength_weights": dict(config.assertion_strength_weights),
        "metric_weights": dict(config.metric_weights),
    }


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, RGBAStatus):
        return value.value
    if isinstance(value, dict):
        return {str(k): _sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(v) for v in value]
    return value


def _build_metric(
    *,
    status: RGBAStatus,
    value: float | None = None,
    confidence: float | None = None,
    reasons: list[str] | None = None,
    trace: dict[str, Any] | None = None,
) -> RGBAMetric:
    return RGBAMetric(
        status=status,
        value=value if status == RGBAStatus.OK else None,
        confidence=confidence if status == RGBAStatus.OK else None,
        reasons=reasons or [],
        trace=trace or {},
    )


def aggregate_rgba_audit(
    *,
    claim_audits: list[dict[str, Any]] | list[ClaimAudit],
    evidence_audits: list[dict[str, Any]] | list[EvidenceAudit],
    sources: list[dict[str, Any]] | None = None,
    trace_context: dict[str, Any] | None = None,
    audit_errors: dict[str, Any] | None = None,
    config: RGBAAuditConfig | None = None,
) -> RGBAResult:
    config = config or default_rgba_audit_config()
    sources = sources or []
    trace_context = trace_context or {}

    claim_audits_list = _coerce_claim_audits(list(claim_audits))
    evidence_audits_list = _coerce_evidence_audits(list(evidence_audits))

    # Robust check for errors: only TRUE if there's actually a non-empty error dict
    has_audit_errors = any(bool(v) for v in audit_errors.values()) if audit_errors else False
    has_claim_audits = bool(claim_audits_list)
    has_evidence_audits = bool(evidence_audits_list)

    clusters = cluster_sources(sources, config) if sources else []
    cluster_sizes: dict[str, int] = {}
    for cluster in clusters:
        for sid in cluster.source_ids:
            cluster_sizes[str(sid)] = cluster.size

    Trace.event(
        "rgba_audit.clustering.summary",
        {
            "cluster_count": len(clusters),
            "cluster_sizes": [cluster.size for cluster in clusters],
        },
    )

    sources_by_id = {str(src.get("source_id")): src for src in sources if isinstance(src, dict)}

    support_mass = 0.0
    refute_mass = 0.0
    per_claim: dict[str, dict[str, float]] = {}

    for audit in evidence_audits_list:
        source_meta = sources_by_id.get(audit.source_id)
        cluster_size = cluster_sizes.get(audit.source_id, 1)

        stance_weight = config.stance_weights.get(audit.stance, 0.0)
        if stance_weight <= 0:
            continue

        directness_weight = config.directness_weights.get(audit.directness, 0.5)
        specificity_weight = config.specificity_weights.get(audit.specificity, 0.5)
        quote_weight = config.quote_integrity_weights.get(audit.quote_integrity, 0.5)
        novelty_weight = config.novelty_weights.get(audit.novelty_vs_copy, 0.8)
        source_prior = _source_reliability(source_meta, config)

        raw_strength = (
            stance_weight
            * directness_weight
            * specificity_weight
            * quote_weight
            * novelty_weight
            * audit.extraction_confidence
            * audit.audit_confidence
            * source_prior
        )
        raw_strength = max(0.0, min(1.0, raw_strength))
        strength = config.c0 + (config.c1 - config.c0) * raw_strength
        strength = strength / max(cluster_size, 1)

        claim_id = audit.claim_id
        per_claim.setdefault(claim_id, {"support": 0.0, "refute": 0.0})

        if audit.stance == "support":
            support_mass += strength
            per_claim[claim_id]["support"] += strength
        elif audit.stance == "refute":
            refute_mass += strength
            per_claim[claim_id]["refute"] += strength

    for claim_id, totals in per_claim.items():
        Trace.event(
            "rgba_audit.claim_summary",
            {
                "claim_id": claim_id,
                "support_mass": round(totals["support"], 4),
                "refute_mass": round(totals["refute"], 4),
            },
        )

    evidence_strength_total = support_mass + refute_mass

    g_trace = {
        "support_mass": round(support_mass, 4),
        "refute_mass": round(refute_mass, 4),
        "evidence_count": len(evidence_audits_list),
        "cluster_count": len(clusters),
    }
    g_reasons: list[str] = []

    if has_audit_errors:
        g_status = RGBAStatus.PIPELINE_ERROR
        g_reasons.append("audit_error")
        g_value = None
        g_confidence = None
    elif not has_evidence_audits or evidence_strength_total <= 0:
        g_status = RGBAStatus.INSUFFICIENT_EVIDENCE
        g_reasons.append("no_usable_evidence")
        g_value = None
        g_confidence = None
    elif (
        support_mass >= config.conflict_threshold
        and refute_mass >= config.conflict_threshold
    ):
        g_status = RGBAStatus.CONFLICTING_EVIDENCE
        g_reasons.append("conflicting_evidence")
        g_value = None
        g_confidence = None
    else:
        g_status = RGBAStatus.OK
        g_value = support_mass / max(evidence_strength_total, 1e-6)
        g_confidence = min(1.0, evidence_strength_total)

    g_metric = _build_metric(
        status=g_status,
        value=g_value,
        confidence=g_confidence,
        reasons=g_reasons,
        trace=g_trace,
    )

    b_trace = {
        "assertion_strength_avg": 0.0,
        "evidence_strength": round(evidence_strength_total, 4),
    }
    b_reasons: list[str] = []

    if has_audit_errors:
        b_metric = _build_metric(
            status=RGBAStatus.PIPELINE_ERROR,
            reasons=["audit_error"],
            trace=b_trace,
        )
    elif not has_claim_audits:
        b_metric = _build_metric(
            status=RGBAStatus.PIPELINE_ERROR,
            reasons=["missing_claim_audits"],
            trace=b_trace,
        )
    elif not has_evidence_audits or evidence_strength_total <= 0:
        b_metric = _build_metric(
            status=RGBAStatus.INSUFFICIENT_EVIDENCE,
            reasons=["no_usable_evidence"],
            trace=b_trace,
        )
    else:
        expected_strength = sum(
            config.assertion_strength_weights.get(audit.assertion_strength, 0.5)
            for audit in claim_audits_list
        ) / max(len(claim_audits_list), 1)
        b_trace["assertion_strength_avg"] = round(expected_strength, 4)
        mismatch_ratio = evidence_strength_total / max(expected_strength, 1e-6)
        b_value = min(1.0, mismatch_ratio)
        b_metric = _build_metric(
            status=RGBAStatus.OK,
            value=b_value,
            confidence=min(1.0, evidence_strength_total),
            reasons=b_reasons,
            trace=b_trace,
        )

    a_trace = {"trace_event_count": 0}
    if not trace_context:
        a_metric = _build_metric(
            status=RGBAStatus.PIPELINE_ERROR,
            reasons=["missing_trace"],
            trace=a_trace,
        )
    else:
        event_count = len(trace_context.get("events", [])) if isinstance(trace_context, dict) else 0
        a_trace["trace_event_count"] = event_count
        a_metric = _build_metric(
            status=RGBAStatus.OK,
            value=1.0 if event_count > 0 else 0.7,
            confidence=1.0,
            reasons=[],
            trace=a_trace,
        )

    r_trace = {"risk_facet_count": 0}
    if not has_claim_audits:
        r_metric = _build_metric(
            status=RGBAStatus.INSUFFICIENT_EVIDENCE,
            reasons=["missing_claim_audits"],
            trace=r_trace,
        )
    else:
        risk_count = sum(len(audit.risk_facets) for audit in claim_audits_list)
        r_trace["risk_facet_count"] = risk_count
        avg_confidence = sum(audit.audit_confidence for audit in claim_audits_list) / max(
            len(claim_audits_list), 1
        )
        r_value = min(1.0, config.risk_weight * (0.1 + 0.1 * risk_count))
        r_metric = _build_metric(
            status=RGBAStatus.OK,
            value=r_value,
            confidence=avg_confidence,
            reasons=[],
            trace=r_trace,
        )

    summary_trace = {
        "claim_count": len(claim_audits_list),
        "evidence_count": len(evidence_audits_list),
        "support_mass": round(support_mass, 4),
        "refute_mass": round(refute_mass, 4),
        "cluster_count": len(clusters),
        "config": _config_snapshot(config),
    }
    if audit_errors:
        summary_trace["audit_errors"] = _sanitize_payload(audit_errors)

    Trace.event(
        "rgba_audit.run_summary",
        {
            "status": {
                "R": r_metric.status.value,
                "G": g_metric.status.value,
                "B": b_metric.status.value,
                "A": a_metric.status.value,
            },
            "claim_count": len(claim_audits_list),
            "evidence_count": len(evidence_audits_list),
        },
    )

    return RGBAResult(
        R=r_metric,
        G=g_metric,
        B=b_metric,
        A=a_metric,
        global_reasons=[],
        summary_trace=summary_trace,
    )
