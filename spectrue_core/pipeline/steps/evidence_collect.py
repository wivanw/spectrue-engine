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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.contracts import (
    EVIDENCE_INDEX_KEY,
    RETRIEVAL_ITEMS_KEY,
    SEARCH_PLAN_KEY,
    EvidenceIndex,
    EvidenceItem,
    EvidencePackContract,
    RetrievalItem,
)
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.pipeline.pipeline_evidence import (
    EvidenceFlowInput,
    collect_evidence,
)

logger = logging.getLogger(__name__)


def _group_sources_by_claim(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        claim_id = source.get("claim_id")
        if not claim_id:
            continue
        grouped.setdefault(str(claim_id), []).append(source)
    return grouped


def _build_evidence_items(raw_items: list[dict[str, Any]]) -> tuple[EvidenceItem, ...]:
    items: list[EvidenceItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        items.append(
            EvidenceItem(
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
            )
        )
    return tuple(items)


def _build_evidence_stats(items: tuple[EvidenceItem, ...]) -> dict[str, Any]:
    n_total = len(items)
    n_quotes = sum(1 for item in items if item.quote)
    n_support = sum(1 for item in items if (item.stance or "").upper() == "SUPPORT")
    n_refute = sum(1 for item in items if (item.stance or "").upper() == "REFUTE")
    n_high_trust = sum(
        1 for item in items if (item.tier or "").upper() in {"A", "A'", "B"}
    )
    return {
        "n_total": n_total,
        "n_quotes": n_quotes,
        "n_support": n_support,
        "n_refute": n_refute,
        "n_high_trust": n_high_trust,
    }


def _build_trace(ctx: PipelineContext, claim_id: str | None = None) -> dict[str, Any]:
    execution_state = ctx.get_extra("execution_state")
    if execution_state is None:
        return {}
    if claim_id and hasattr(execution_state, "claim_states"):
        claim_states = getattr(execution_state, "claim_states", {})
        claim_state = claim_states.get(claim_id)
        if claim_state and hasattr(claim_state, "to_dict"):
            return claim_state.to_dict()
    if hasattr(execution_state, "to_dict"):
        return execution_state.to_dict()
    return {}


@dataclass
class EvidenceCollectStep:
    """
    Collect evidence packs without invoking any judging.

    This step prepares evidence for downstream judging steps and produces
    a stable EvidenceIndex contract.
    """

    agent: Any  # FactCheckerAgent
    search_mgr: Any  # SearchManager
    include_global_pack: bool = True
    name: str = "evidence_collect"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        try:
            # Skip if verdict already exists (oracle hit or gating rejection)
            if ctx.get_extra("gating_rejected") or ctx.get_extra("oracle_hit"):
                return ctx

            claims = ctx.claims
            sources = ctx.sources
            progress_callback = ctx.get_extra("progress_callback")

            retrieval_items = ctx.get_extra(RETRIEVAL_ITEMS_KEY)
            global_items: list[dict[str, Any]] = []
            by_claim_items: dict[str, list[dict[str, Any]]] = {}

            if isinstance(retrieval_items, dict):
                raw_global = retrieval_items.get("global", []) if self.include_global_pack else []
                raw_by_claim = retrieval_items.get("by_claim", {})
                for item in raw_global or []:
                    if isinstance(item, RetrievalItem):
                        global_items.append(item.to_payload())
                    elif isinstance(item, dict):
                        global_items.append(item)
                if isinstance(raw_by_claim, dict):
                    for claim_id, items in raw_by_claim.items():
                        if not isinstance(items, (list, tuple)):
                            continue
                        by_claim_items[str(claim_id)] = []
                        for item in items:
                            if isinstance(item, RetrievalItem):
                                by_claim_items[str(claim_id)].append(item.to_payload())
                            elif isinstance(item, dict):
                                by_claim_items[str(claim_id)].append(item)

            if global_items or by_claim_items:
                flat_by_claim = [item for items in by_claim_items.values() for item in items]
                if self.include_global_pack:
                    sources = global_items + flat_by_claim
                else:
                    # Deep mode: ignore any global items (avoid cross-claim/global leakage)
                    sources = flat_by_claim
            else:
                by_claim_items = _group_sources_by_claim(sources)

            inp = EvidenceFlowInput(
                fact=ctx.get_extra("prepared_fact", ""),
                original_fact=ctx.get_extra("original_fact", ""),
                lang=ctx.lang,
                content_lang=ctx.lang,
                gpt_model=ctx.gpt_model,
                search_type=ctx.search_type,
                progress_callback=progress_callback,
            )

            from spectrue_core.verification.evidence.evidence import build_evidence_pack

            collection = await collect_evidence(
                agent=self.agent,
                search_mgr=self.search_mgr,
                build_evidence_pack=build_evidence_pack,
                calibration_registry=None,
                inp=inp,
                claims=claims,
                sources=sources,
            )

            evidence_by_claim = by_claim_items or _group_sources_by_claim(collection.sources)

            pack_items = list(collection.pack.get("items", [])) if isinstance(collection.pack, dict) else []
            pack_contract = None
            if self.include_global_pack:
                pack_contract = EvidencePackContract(
                    items=_build_evidence_items(pack_items),
                    stats=dict(collection.pack.get("stats", {})) if isinstance(collection.pack, dict) else {},
                    trace=_build_trace(ctx),
                )

            by_claim_contract: dict[str, EvidencePackContract] = {}
            for claim_id, raw_items in evidence_by_claim.items():
                items = _build_evidence_items(raw_items)
                by_claim_contract[claim_id] = EvidencePackContract(
                    items=items,
                    stats=_build_evidence_stats(items),
                    trace=_build_trace(ctx, claim_id),
                )

            claim_ids = []
            for claim in claims or []:
                if not isinstance(claim, dict):
                    continue
                cid = claim.get("id") or claim.get("claim_id")
                if cid:
                    claim_ids.append(str(cid))
            missing_claims = [cid for cid in claim_ids if cid not in by_claim_contract]

            evidence_index = EvidenceIndex(
                by_claim_id=by_claim_contract,
                global_pack=pack_contract,
                stats={
                    "claims_total": len(claim_ids),
                    "claims_with_evidence": len(by_claim_contract),
                    "missing_claims": len(missing_claims),
                },
                trace={
                    "plan_id": getattr(ctx.get_extra(SEARCH_PLAN_KEY), "plan_id", None),
                },
                missing_claims=tuple(missing_claims),
            )

            Trace.event(
                "evidence_collect.completed",
                {
                    "claims_with_evidence": len(by_claim_contract),
                    "missing_claims": len(missing_claims),
                },
            )

            execution_state = ctx.get_extra("execution_state")
            if execution_state is not None and hasattr(execution_state, "claim_states"):
                ctx = ctx.set_extra("execution_states", execution_state.claim_states)

            # CRITICAL: Only populate ctx.evidence/ctx.sources for standard mode
            # Deep mode must NOT have global evidence pollution
            if self.include_global_pack:
                # Standard mode: update ctx.evidence and ctx.sources with global pack
                return (
                    ctx.with_update(sources=collection.sources, evidence=collection.pack)
                    .set_extra("evidence_collection", collection)
                    .set_extra("evidence_by_claim", evidence_by_claim)
                    .set_extra(EVIDENCE_INDEX_KEY, evidence_index)
                )
            else:
                # Deep mode: only set EvidenceIndex, do NOT pollute ctx.evidence
                Trace.event(
                    "evidence_collect.deep_mode_no_global",
                    {"claims_count": len(by_claim_contract)},
                )
                return (
                    ctx.set_extra("evidence_collection", collection)
                    .set_extra("evidence_by_claim", evidence_by_claim)
                    .set_extra(EVIDENCE_INDEX_KEY, evidence_index)
                )

        except Exception as e:
            logger.exception("[EvidenceCollectStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
