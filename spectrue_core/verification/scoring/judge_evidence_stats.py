# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from typing import Any


def build_judge_evidence_stats(
    *,
    claim_id: str,
    corroboration_by_claim: dict[str, dict[str, Any]] | None,
    evidence_stats_by_claim: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """
    Build a compact, structured evidence_stats object for the judge.
    This is purely observational/diagnostic:
    - it must NOT be used as a substitute for evidence spans
    - it helps the judge reason about redundancy vs independence and coverage
    """
    out: dict[str, Any] = {}

    corr = (corroboration_by_claim or {}).get(claim_id) if isinstance(corroboration_by_claim, dict) else None
    est = (evidence_stats_by_claim or {}).get(claim_id) if isinstance(evidence_stats_by_claim, dict) else None

    if isinstance(est, dict):
        out["sources_observed"] = est.get("sources_observed", 0)
        out["unique_urls"] = est.get("unique_urls", 0)
        out["unique_domains"] = est.get("unique_domains", 0)
        out["direct_anchors"] = est.get("direct_anchors", 0)
        out["covered_slots"] = est.get("covered_slots", 0)
        out["transferred"] = est.get("transferred", 0)

    if isinstance(corr, dict):
        out["precision_publishers_support"] = corr.get("precision_publishers_support", 0)
        out["precision_publishers_refute"] = corr.get("precision_publishers_refute", 0)
        out["corroboration_clusters_support"] = corr.get("corroboration_clusters_support", 0)
        out["corroboration_clusters_refute"] = corr.get("corroboration_clusters_refute", 0)
        out["unique_publishers_total"] = corr.get("unique_publishers_total", 0)
        out["exact_content_groups"] = corr.get("exact_content_groups", 0)

    return out
