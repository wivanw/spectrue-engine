# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tiered retrieval for deep_v2: trusted domains first, the general tier only after stance.

Sufficiency (`check_sufficiency_for_claim`) needs quotes and stance to reach SUFFICIENT; both exist
only after stance annotation. Judged on raw search results it was never met and every query was
searched twice (trace 2026-10-18_20-22-24: 38 queries -> 76 searches). Judged on read pages without
quotes it was still rarely met (51 of 61 queries went general, trace 2026-10-18 rerun). So the
decision is taken after stance: a cluster gets the general tier only if its stance-annotated
evidence is still insufficient. The second round reuses the same retrieval steps on a sub-context
holding only those clusters' plans; its results are merged into the main context here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from spectrue_core.use_cases.claims.sufficiency import SufficiencyStatus, check_sufficiency_for_claim


def _claim_id(c: Any) -> str:
    return str((c or {}).get("id") or (c or {}).get("claim_id") or "") if isinstance(c, dict) else ""


def clusters_insufficient_after_stance(cluster_plans: list[dict], sources: Iterable[dict]) -> dict[str, Any]:
    """cluster_id -> SufficiencyResult for clusters whose stance-annotated evidence is insufficient."""
    by_claim: dict[str, list[dict]] = {}
    for s in sources or []:
        if isinstance(s, dict) and s.get("claim_id"):
            by_claim.setdefault(str(s["claim_id"]), []).append(s)
    insufficient: dict[str, Any] = {}
    for plan in cluster_plans:
        claims = [c for c in (plan.get("claims") or []) if _claim_id(c)]
        if not claims or not plan.get("search_queries"):
            continue
        pooled = [s for c in claims for s in by_claim.get(_claim_id(c), [])]
        result = check_sufficiency_for_claim(claims[0], pooled)
        if result.status != SufficiencyStatus.SUFFICIENT:
            insufficient[str(plan.get("cluster_id") or "cluster")] = result
    return insufficient


def merge_sources(main: Iterable[dict], extra: Iterable[dict]) -> list[dict]:
    """Main sources plus the round's sources; a (claim, url) pair already present is not duplicated."""
    out = [s for s in main or [] if isinstance(s, dict)]
    seen = {(str(s.get("claim_id") or ""), str(s.get("url") or "")) for s in out}
    for s in extra or []:
        key = (str((s or {}).get("claim_id") or ""), str((s or {}).get("url") or ""))
        if isinstance(s, dict) and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def merge_by_claim(main: Mapping[str, list] | None, extra: Mapping[str, list] | None) -> dict[str, list]:
    out = {k: list(v) for k, v in (main or {}).items()}
    for cid, items in (extra or {}).items():
        out[cid] = merge_sources(out.get(cid, []), items)
    return out


def merge_index(main: Any, extra: Any) -> Any:
    """EvidenceIndex union per claim (items appended, missing_claims recomputed)."""
    if main is None or extra is None:
        return main or extra
    from dataclasses import replace

    by_claim = dict(main.by_claim_id)
    for cid, pack in extra.by_claim_id.items():
        if cid in by_claim:
            by_claim[cid] = replace(by_claim[cid], items=tuple(by_claim[cid].items) + tuple(pack.items))
        else:
            by_claim[cid] = pack
    missing = tuple(c for c in main.missing_claims if not (by_claim.get(c) and by_claim[c].items))
    return replace(main, by_claim_id=by_claim, missing_claims=missing)


MERGED_DICT_EXTRAS = ("evidence_docs", "evidence_doc_meta", "evidence_item_meta", "cluster_evidence_docs")


def merge_round(main_ctx: Any, round_ctx: Any) -> Any:
    """The main context with the second round's evidence merged in (sources, per-claim maps, collection, index)."""
    from spectrue_core.domain.verification.verdict.model import EvidenceFlowInput
    from spectrue_core.pipeline.contracts import EVIDENCE_INDEX_KEY
    from spectrue_core.pipeline.evidence_flow import rebuild_evidence_pack
    from spectrue_core.utils.evidence import build_evidence_pack

    sources = merge_sources(main_ctx.sources, round_ctx.sources)
    ctx = main_ctx.with_update(sources=sources)
    for key in MERGED_DICT_EXTRAS:
        ctx = ctx.set_extra(key, {**(main_ctx.get_extra(key) or {}), **(round_ctx.get_extra(key) or {})})
    ctx = ctx.set_extra("audit_sources", list(main_ctx.get_extra("audit_sources") or [])
                        + list(round_ctx.get_extra("audit_sources") or []))
    ctx = ctx.set_extra("evidence_by_claim", merge_by_claim(main_ctx.get_extra("evidence_by_claim"),
                                                            round_ctx.get_extra("evidence_by_claim")))
    ctx = ctx.set_extra(EVIDENCE_INDEX_KEY, merge_index(main_ctx.get_extra(EVIDENCE_INDEX_KEY),
                                                        round_ctx.get_extra(EVIDENCE_INDEX_KEY)))
    collection = main_ctx.get_extra("evidence_collection")
    if collection is not None:
        inp = EvidenceFlowInput(fact=main_ctx.get_extra("prepared_fact", ""), original_fact=main_ctx.get_extra("original_fact", ""),
                                lang=main_ctx.lang, content_lang=main_ctx.lang, analysis_mode=main_ctx.mode.api_analysis_mode,
                                progress_callback=None)
        collection = rebuild_evidence_pack(build_evidence_pack=build_evidence_pack, collection=collection,
                                           clustered_results=sources, sources_override=sources, inp=inp)
        ctx = ctx.with_update(evidence=collection.pack).set_extra("evidence_collection", collection)
    return ctx
