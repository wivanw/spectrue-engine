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

from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.utils.trace import Trace


def _stance_class(s: dict[str, Any]) -> str:
    return str(s.get("stance") or "").upper()


def _is_precision(s: dict[str, Any]) -> bool:
    # Precision evidence = has direct anchor
    return bool(s.get("quote_span") or s.get("contradiction_span") or s.get("quote"))


@dataclass
class EvidenceCorroborationStep(Step):
    """
    Compute per-claim corroboration counters using:
    - precise confirmations: unique publishers with SUPPORT/REFUTE and direct anchors
    - corroboration confirmations: unique similar clusters with SUPPORT/REFUTE (any anchor)
    - exact duplicate count (informational)
    """
    weight: float = 1.0

    name: str = "evidence_corroboration"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.api_analysis_mode != AnalysisMode.DEEP_V2:
            return ctx

        sources = ctx.sources or []
        claims = ctx.claims or []
        if not sources or not claims:
            return ctx

        # Prefer evidence_by_claim
        by_claim = ctx.get_extra("evidence_by_claim")
        if not isinstance(by_claim, dict):
            by_claim = {}
            for s in sources:
                if not isinstance(s, dict):
                    continue
                cid = s.get("claim_id")
                if not cid:
                    continue
                by_claim.setdefault(str(cid), []).append(s)

        out: dict[str, dict[str, Any]] = {}
        for c in claims:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id") or c.get("claim_id") or "")
            if not cid:
                continue
            items = [x for x in by_claim.get(cid, []) if isinstance(x, dict)]

            pub_support = set()
            pub_refute = set()
            clu_support = set()
            clu_refute = set()
            exact_hashes = set()
            pubs_all = set()

            for s in items:
                st = _stance_class(s)
                pub = str(s.get("publisher_id") or "")
                if pub:
                    pubs_all.add(pub)
                ch = str(s.get("content_hash") or "")
                if ch:
                    exact_hashes.add(ch)

                # corroboration clusters count any support/refute
                scid = str(s.get("similar_cluster_id") or "")
                if st in {"SUPPORT"} and scid:
                    clu_support.add(scid)
                if st in {"REFUTE"} and scid:
                    clu_refute.add(scid)

                # precision publishers count only direct anchors
                if _is_precision(s):
                    if st in {"SUPPORT"} and pub:
                        pub_support.add(pub)
                    if st in {"REFUTE"} and pub:
                        pub_refute.add(pub)

            out[cid] = {
                "precision_publishers_support": len(pub_support),
                "precision_publishers_refute": len(pub_refute),
                "corroboration_clusters_support": len(clu_support),
                "corroboration_clusters_refute": len(clu_refute),
                "unique_publishers_total": len(pubs_all),
                "exact_content_groups": len(exact_hashes),
                "evidence_items": len(items),
            }

        Trace.event(
            "evidence_corroboration.completed",
            {
                "claims": len(out),
                "avg_precise_support": round(
                    sum(v["precision_publishers_support"] for v in out.values()) / max(1, len(out)),
                    3,
                ),
                "avg_corr_support": round(
                    sum(v["corroboration_clusters_support"] for v in out.values()) / max(1, len(out)),
                    3,
                ),
            },
        )

        return ctx.set_extra("corroboration_by_claim", out)
