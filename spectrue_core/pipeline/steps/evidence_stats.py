# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.retrieval.fixed_pipeline import normalize_url


def _norm_domain(src: dict[str, Any]) -> str:
    d = src.get("domain") or ""
    d = str(d).strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d


def _has_direct_anchor(src: dict[str, Any]) -> bool:
    return bool(src.get("quote_span") or src.get("contradiction_span") or src.get("quote"))


def _collect_slots(src: dict[str, Any]) -> set[str]:
    slots = set()
    covers = src.get("covers")
    if isinstance(covers, list):
        for x in covers:
            s = str(x).strip().lower()
            if s:
                slots.add(s)
    return slots


def _deterministic_A(stats: dict[str, Any], bayesian_p: float = 0.5) -> float:
    """
    Deterministic explainability score in [0,1] based on observable artifacts:
    - direct anchors (quote_span/contradiction_span/quote)
    - number of unique domains (diversity)
    - slot coverage (covers[])
    - bayesian sufficiency (proof density)
    This is NOT a verdict; it's a traceability score.
    """
    direct = float(stats.get("direct_anchors", 0))
    uniq = float(stats.get("unique_domains", 0))
    slots = float(stats.get("covered_slots", 0))
    # Saturating functions (no thresholds): 1 - exp(-k*x)
    a1 = 1.0 - math.exp(-0.9 * direct)
    a2 = 1.0 - math.exp(-0.6 * uniq)
    a3 = 1.0 - math.exp(-0.4 * slots)
    a4 = max(0.0, min(1.0, float(bayesian_p)))
    # Weighted blend (fixed, documented)
    # Bayesian consensus now accounts for 30% of the deterministic signal
    out = 0.40 * a1 + 0.15 * a2 + 0.15 * a3 + 0.30 * a4
    return max(0.0, min(1.0, out))


@dataclass
class EvidenceStatsStep(Step):
    """
    Build per-claim EvidenceStats from already selected sources.
    This decouples explainability (A) from BudgetState and fetch/extract counts.
    """
    weight: float = 1.0
    name: str = "evidence_stats"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.api_analysis_mode != AnalysisMode.DEEP_V2:
            return ctx

        sources = ctx.sources or []
        claims = ctx.claims or []
        if not sources or not claims:
            return ctx

        # Prefer evidence_by_claim if present (built by spillover steps)
        by_claim = ctx.get_extra("evidence_by_claim")
        if not isinstance(by_claim, dict):
            # fallback grouping
            by_claim = {}
            for s in sources:
                if not isinstance(s, dict):
                    continue
                cid = s.get("claim_id")
                if not cid:
                    continue
                by_claim.setdefault(str(cid), []).append(s)

        stats_by_claim: dict[str, dict[str, Any]] = {}
        for c in claims:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id") or c.get("claim_id") or "")
            if not cid:
                continue
            items = [x for x in by_claim.get(cid, []) if isinstance(x, dict)]

            unique_urls = set()
            unique_domains = set()
            direct_anchors = 0
            transferred = 0
            best_tier: str | None = None
            slots = set()

            def _rank(t):
                return {"D": 1, "C": 2, "B": 3, "A'": 3, "A": 4}.get(str(t).strip().upper(), 0)

            for s in items:
                url = s.get("url")
                if url:
                    try:
                        unique_urls.add(normalize_url(str(url)))
                    except Exception:
                        unique_urls.add(str(url))
                d = _norm_domain(s)
                if d:
                    unique_domains.add(d)
                if _has_direct_anchor(s):
                    direct_anchors += 1
                if s.get("provenance") == "transferred":
                    transferred += 1
                slots |= _collect_slots(s)
                
                tier = s.get("tier") or s.get("source_tier")
                if tier and (best_tier is None or _rank(tier) > _rank(best_tier)):
                    best_tier = str(tier).strip().upper()

            # Retrieve Bayesian sufficiency from cluster search
            cluster_map = ctx.get_extra("cluster_map") or {}
            cluster_sufficiency = ctx.get_extra("cluster_sufficiency") or {}
            cluster_id = cluster_map.get(cid)
            bayesian_p = cluster_sufficiency.get(cluster_id, 0.5) if cluster_id else 0.5

            stats = {
                "sources_observed": len(items),
                "unique_urls": len(unique_urls),
                "unique_domains": len(unique_domains),
                "direct_anchors": direct_anchors,
                "covered_slots": len(slots),
                "transferred": transferred,
                "bayesian_p": bayesian_p,
                "best_tier": best_tier,
            }
            stats["A_deterministic"] = _deterministic_A(stats, bayesian_p=bayesian_p)
            stats_by_claim[cid] = stats

        Trace.event(
            "evidence_stats.completed",
            {
                "claims": len(stats_by_claim),
                "avg_sources": round(
                    sum(v["sources_observed"] for v in stats_by_claim.values()) / max(1, len(stats_by_claim)),
                    3,
                ),
                "avg_A_det": round(
                    sum(v["A_deterministic"] for v in stats_by_claim.values()) / max(1, len(stats_by_claim)),
                    3,
                ),
            },
        )

        return ctx.set_extra("evidence_stats_by_claim", stats_by_claim)
