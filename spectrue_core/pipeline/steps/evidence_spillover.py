# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectrue_core.use_cases.evidence.spillover import run_spillover
from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.runtime_config import DeepV2Config
from spectrue_core.utils.trace import Trace
from spectrue_core.utils.retrieval_urls import normalize_url


@dataclass
class EvidenceSpilloverStep(Step):
    """
    Share compatible evidence between claims inside the same claim cluster.

    Deep v2 only:
    - No new search
    - No new LLM calls
    - Deterministic routing + provenance marking
    """

    config: Any
    name: str = "evidence_spillover"
    weight: float = 2.0



    async def run(self, ctx: PipelineContext) -> PipelineContext:
        sources = ctx.sources or []
        claims = ctx.claims or []
        if not sources or not claims:
            return ctx

        cluster_map: dict[str, str] = ctx.get_extra("cluster_map") or {}
        if not cluster_map:
            Trace.event("evidence_spillover.skipped", {"reason": "no_cluster_map"})
            return ctx

        runtime = getattr(self.config, "runtime", None)
        deep_v2_cfg = getattr(runtime, AnalysisMode.DEEP_V2.value, DeepV2Config())
        top_k = int(getattr(deep_v2_cfg, "corroboration_top_k", 3) or 3)

        # Prefer existing grouped mapping if present
        by_claim = ctx.get_extra("evidence_by_claim")
        if not isinstance(by_claim, dict):
            by_claim = None

        result = run_spillover(
            sources=sources,
            claims=claims,
            cluster_map=cluster_map,
            top_k=top_k,
            normalize_url=normalize_url,
            evidence_by_claim=by_claim,
        )

        for choice in result.choices:
            Trace.event(
                "evidence_spillover.chosen",
                {
                    "claim_id": choice.claim_id,
                    "cluster_id": choice.cluster_id,
                    "count": choice.count,
                    "urls": choice.urls,
                    "topic_boost_used": choice.topic_boost_used,
                },
            )

        Trace.event(
            "evidence_spillover.completed",
            {
                "transferred": result.transferred_total,
                "touched_claims": result.touched_claims,
                "top_k": top_k,
                "rejections": result.rejections,
            },
        )

        if not result.transferred_items:
            return ctx

        # Build replacement context
        new_ctx = (
            ctx.with_update(sources=result.combined_sources)
            .set_extra("evidence_by_claim", result.evidence_by_claim)
            .set_extra("spillover_transferred", result.transferred_total)
        )

        # 8. M115/M119: Update EvidenceIndex if present
        from spectrue_core.pipeline.contracts import EVIDENCE_INDEX_KEY, EvidenceIndex, EvidencePackContract, EvidenceItem
        old_index: EvidenceIndex | None = ctx.get_extra(EVIDENCE_INDEX_KEY)
        if old_index:
            new_by_claim = dict(old_index.by_claim_id)
            
            # Group transferred by target claim
            transferred_by_cid = result.evidence_by_claim
            
            for cid, raw_shared in transferred_by_cid.items():
                # Build EvidenceItem objects
                shared_items = []
                for raw in raw_shared:
                    if raw.get("provenance") != "transferred":
                        continue
                    shared_items.append(EvidenceItem(
                        url=str(raw.get("url") or raw.get("link") or ""),
                        source_id=raw.get("source_id"),
                        title=raw.get("title"),
                        snippet=raw.get("snippet") or raw.get("content"),
                        quote=raw.get("quote"),
                        provider_score=raw.get("provider_score") or raw.get("score"),
                        sim=raw.get("sim") if raw.get("sim") is not None else raw.get("similarity_score"),
                        stance=raw.get("stance"),
                        relevance=raw.get("relevance"),
                        tier=raw.get("tier") or raw.get("source_tier") or raw.get("evidence_tier"),
                    ))
                
                old_pack = new_by_claim.get(cid)
                if old_pack:
                    # Update existing pack contract (immutable)
                    new_by_claim[cid] = EvidencePackContract(
                        items=tuple(list(old_pack.items) + shared_items),
                        stats=dict(old_pack.stats), # Stats might need recalculation but we prioritize items
                        trace=dict(old_pack.trace),
                    )
                else:
                    # Create new pack contract
                    new_by_claim[cid] = EvidencePackContract(
                        items=tuple(shared_items),
                        stats={"n_total": len(shared_items), "transferred": True},
                        trace={},
                    )
            
            # Replace frozen index
            new_index = EvidenceIndex(
                by_claim_id=new_by_claim,
                global_pack=old_index.global_pack,
                stats=dict(old_index.stats),
                trace=dict(old_index.trace),
                missing_claims=old_index.missing_claims,
            )
            new_ctx = new_ctx.set_extra(EVIDENCE_INDEX_KEY, new_index)

        # 9. Update global EvidencePack (ctx.evidence) if present (Standard mode)
        if ctx.evidence and isinstance(ctx.evidence, dict):
            # In standard mode, evidence is a dict with 'items' key
            ev_items = list(ctx.evidence.get("items", []))
            # Just append transferred items as raw dicts (backward compat)
            ev_items.extend(result.transferred_items)
            new_evidence = {**ctx.evidence, "items": ev_items}
            new_ctx = new_ctx.with_update(evidence=new_evidence)

        return new_ctx
