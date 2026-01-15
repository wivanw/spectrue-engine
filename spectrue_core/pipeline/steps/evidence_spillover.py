# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.runtime_config import DeepV2Config
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.retrieval.fixed_pipeline import normalize_url
from spectrue_core.verification.evidence.slot_maps import (
    merge_covers,
    required_slots_for_verification_target,
    slots_from_assertion_key,
)


def _group_sources_by_claim(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in sources:
        if not isinstance(s, dict):
            continue
        cid = s.get("claim_id")
        if not cid:
            continue
        grouped.setdefault(str(cid), []).append(s)
    return grouped


def _claim_assertion_keys(claim: dict[str, Any]) -> tuple[set[str], set[str]]:
    """
    Return (fact_keys, context_keys) from claim.assertions[] if present.
    Deterministic and schema-driven.
    """
    fact: set[str] = set()
    ctx: set[str] = set()
    assertions = claim.get("assertions")
    if not isinstance(assertions, list):
        return fact, ctx

    for a in assertions:
        if not isinstance(a, dict):
            continue
        key = a.get("key")
        if not key:
            continue
        dim = str(a.get("dimension") or "FACT").upper()
        if dim == "CONTEXT":
            ctx.add(str(key))
        else:
            fact.add(str(key))
    return fact, ctx


def _is_transfer_candidate(src: dict[str, Any]) -> bool:
    """
    Conservative: transfer only evidence with an explainability anchor.
    No text heuristics.
    """
    stance = str(src.get("stance") or "").upper()
    if stance in {"IRRELEVANT"}:
        return False
    # Prevent cascade: transferred items must not be used as donors.
    if src.get("provenance") == "transferred":
        return False
    # Require an anchor for downstream explainability/judge
    if src.get("quote") or src.get("quote_span") or src.get("contradiction_span"):
        return True
    return False


def _compatible_for_claim(src: dict[str, Any], fact_keys: set[str], context_keys: set[str]) -> bool:
    """
    Deterministic compatibility using:
    - assertion_key (evidence → which assertion it applies to)
    - stance class (SUPPORT/REFUTE vs CONTEXT/MENTION)
    - claim assertions dimension (FACT/CONTEXT)
    """
    akey = str(src.get("assertion_key") or "")
    stance = str(src.get("stance") or "").upper()

    # Legacy whole-claim evidence: allow (routing v2 keeps it conservative elsewhere)
    if not akey:
        return True

    if stance in {"SUPPORT", "REFUTE", "MIXED"}:
        return akey in fact_keys

    if stance in {"CONTEXT", "MENTION"}:
        return akey in context_keys

    return False


def _extract_verification_target(claim: dict[str, Any]) -> str:
    """
    Extract verification_target from claim metadata (schema-driven).
    Handles both flattened and nested representations.
    """
    md = claim.get("metadata")
    if isinstance(md, dict) and md.get("verification_target"):
        return str(md.get("verification_target"))
    if claim.get("verification_target"):
        return str(claim.get("verification_target"))
    return ""


def _covers_ok_for_claim(src: dict[str, Any], claim: dict[str, Any]) -> bool:
    """
    Deterministic compatibility using slots:
    required_slots(verification_target) must intersect evidence covers.
    """
    vt = _extract_verification_target(claim)
    required = required_slots_for_verification_target(vt)
    if not required:
        return True

    akey = str(src.get("assertion_key") or "")
    derived = slots_from_assertion_key(akey)
    covers = merge_covers(src.get("covers"), derived)

    return bool(covers & required)


def _score_for_transfer(src: dict[str, Any]) -> float:
    """
    Deterministic ranking using already computed fields.
    Avoids new heuristics or LLM calls.
    """
    try:
        rel = float(src.get("relevance_score", 0.0) or 0.0)
    except Exception:
        rel = 0.0
    stance = str(src.get("stance") or "").upper()
    stance_boost = 0.05 if stance in {"SUPPORT", "REFUTE"} else 0.0
    quote_boost = 0.05 if (src.get("quote") or src.get("quote_span") or src.get("contradiction_span")) else 0.0
    return rel + stance_boost + quote_boost


def _stable_key(src: dict[str, Any]) -> tuple[str, str]:
    """
    Deterministic tie-breaker key.
    We use normalized URL (primary) + domain (secondary).
    """
    url = src.get("url") or ""
    dom = src.get("domain") or ""
    try:
        nurl = normalize_url(str(url)) if url else ""
    except Exception:
        nurl = str(url)
    return (nurl, str(dom))


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

    @property
    def name(self) -> str:
        return "evidence_spillover"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.api_analysis_mode != AnalysisMode.DEEP_V2:
            return ctx

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
            by_claim = _group_sources_by_claim([s for s in sources if isinstance(s, dict)])

        # claim_id -> claim dict (avoid mutation)
        claim_lookup: dict[str, dict[str, Any]] = {}
        for idx, c in enumerate(claims):
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id") or c.get("claim_id") or f"c{idx + 1}")
            claim_lookup[cid] = c

        # cluster_id -> claim_ids
        cluster_to_ids: dict[str, list[str]] = {}
        for cid, clid in cluster_map.items():
            cluster_to_ids.setdefault(clid, []).append(cid)

        transferred_items: list[dict[str, Any]] = []
        transferred_total = 0
        touched_claims = 0
        rejected_slot = 0

        for target_id, target_claim in claim_lookup.items():
            clid = cluster_map.get(target_id)
            if not clid:
                continue
            peers = cluster_to_ids.get(clid, [])
            if len(peers) <= 1:
                continue

            fact_keys, context_keys = _claim_assertion_keys(target_claim)

            # Existing URLs for dedup
            existing_urls: set[str] = set()
            for s in by_claim.get(target_id, []):
                url = s.get("url")
                if url:
                    existing_urls.add(normalize_url(str(url)))

            candidates: list[tuple[float, str, dict[str, Any]]] = []
            for peer_id in peers:
                if peer_id == target_id:
                    continue
                for src in by_claim.get(peer_id, []):
                    if not isinstance(src, dict):
                        continue
                    if not _is_transfer_candidate(src):
                        continue
                    if not _compatible_for_claim(src, fact_keys, context_keys):
                        continue
                    # Compatibility v3: required slots for claim's verification_target
                    if not _covers_ok_for_claim(src, target_claim):
                        rejected_slot += 1
                        continue
                    url = src.get("url")
                    if url and normalize_url(str(url)) in existing_urls:
                        continue
                    candidates.append((_score_for_transfer(src), peer_id, src))

            if not candidates:
                continue

            # Stable deterministic ordering:
            # 1) score desc
            # 2) normalized url asc
            # 3) origin claim id asc
            candidates.sort(
                key=lambda t: (
                    -t[0],
                    _stable_key(t[2])[0],
                    str(t[1]),
                )
            )
            chosen = candidates[:top_k]
            if chosen:
                touched_claims += 1

            chosen_urls = []
            for _, origin_id, src in chosen:
                url = src.get("url")
                if url:
                    existing_urls.add(normalize_url(str(url)))
                    chosen_urls.append(normalize_url(str(url)))
                merged = dict(src)
                merged["claim_id"] = target_id
                merged["provenance"] = "transferred"
                merged["origin_claim_id"] = origin_id
                transferred_items.append(merged)
                transferred_total += 1

            Trace.event(
                "evidence_spillover.chosen",
                {
                    "claim_id": target_id,
                    "cluster_id": clid,
                    "count": len(chosen_urls),
                    "urls": chosen_urls[:5],  # cap
                },
            )

        if not transferred_items:
            Trace.event("evidence_spillover.completed", {"transferred": 0, "touched_claims": 0, "top_k": top_k})
            return ctx

        combined_sources = list(sources) + transferred_items
        by_claim2 = _group_sources_by_claim([s for s in combined_sources if isinstance(s, dict)])

        Trace.event(
            "evidence_spillover.completed",
            {
                "transferred": transferred_total,
                "touched_claims": touched_claims,
                "top_k": top_k,
                "rejected_slot": rejected_slot,
            },
        )

        return (
            ctx.with_update(sources=combined_sources)
            .set_extra("evidence_by_claim", by_claim2)
            .set_extra("spillover_transferred", transferred_total)
        )
