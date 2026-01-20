from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.runtime_config import DeepV2Config
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.pipeline.pipeline_evidence import annotate_evidence_stance
from spectrue_core.verification.retrieval.fixed_pipeline import normalize_url


def _group_by_claim(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = it.get("claim_id")
        if not cid:
            continue
        out.setdefault(str(cid), []).append(it)
    return out


def _pick_top_k_transferred(items: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    # Deterministic ranking: reuse existing relevance_score if present.
    def score(x: dict[str, Any]) -> float:
        try:
            return float(x.get("relevance_score", 0.0) or 0.0)
        except Exception:
            return 0.0

    # Stable deterministic ordering:
    # 1) relevance_score desc
    # 2) normalized url asc
    # 3) origin_claim_id asc
    def stable_sort_key(x: dict[str, Any]) -> tuple[float, str, str]:
        url = x.get("url") or ""
        try:
            nurl = normalize_url(str(url)) if url else ""
        except Exception:
            nurl = str(url)
        oc = str(x.get("origin_claim_id") or "")
        return (-score(x), nurl, oc)

    ranked = sorted(items, key=stable_sort_key)
    out: list[dict[str, Any]] = []
    # Dedup by normalized URL within a claim
    seen: set[str] = set()
    for it in ranked:
        if len(out) >= k:
            break
        url = it.get("url")
        if url:
            nu = normalize_url(str(url))
            if nu in seen:
                continue
            seen.add(nu)
        out.append(it)
    return out


@dataclass
class TransferredStanceAnnotateStep(Step):
    """
    Re-run stance annotation for transferred evidence under the TARGET claim.

    Why:
    - EvidenceSpilloverStep transfers evidence across claims.
    - But stance/quote spans computed under origin claim are not valid for target claim.
    - Without re-annotation, judge may see mismatched evidence and fall to NEI (G=-1).

    Constraints:
    - No new search.
    - Only top-K transferred items per claim.
    """

    agent: Any
    config: Any
    weight: float = 2.0

    name: str = "transferred_stance_annotate"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.api_analysis_mode != AnalysisMode.DEEP_V2:
            return ctx

        sources = ctx.sources or []
        claims = ctx.claims or []
        if not sources or not claims:
            return ctx

        transferred = [
            s for s in sources
            if isinstance(s, dict) and s.get("provenance") == "transferred"
        ]
        if not transferred:
            Trace.event("transferred_stance.skipped", {"reason": "no_transferred"})
            return ctx

        runtime = getattr(self.config, "runtime", None)
        deep_v2_cfg = getattr(runtime, AnalysisMode.DEEP_V2.value, DeepV2Config())
        top_k = int(getattr(deep_v2_cfg, "restace_transferred_top_k", 2) or 2)

        by_claim = _group_by_claim(transferred)

        # Build claim lookup
        claim_lookup: dict[str, dict[str, Any]] = {}
        for idx, c in enumerate(claims):
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id") or c.get("claim_id") or f"c{idx+1}")
            claim_lookup[cid] = c

        updated_map: dict[tuple[str, str], dict[str, Any]] = {}
        total_pairs = 0

        # For each claim, re-annotate only top-K transferred sources
        for cid, items in by_claim.items():
            claim = claim_lookup.get(cid)
            if not claim:
                continue

            chosen = _pick_top_k_transferred(items, top_k)
            if not chosen:
                continue

            Trace.event(
                "transferred_stance.chosen",
                {
                    "claim_id": cid,
                    "count": len(chosen),
                    "urls": [normalize_url(str(x.get("url"))) for x in chosen if x.get("url")][:5],
                },
            )

            # annotate_evidence_stance expects sources + claims (batch) + inp
            # We send a single-claim batch to bind stance to this claim.
            # Create minimal EvidenceFlowInput for stance annotation
            from spectrue_core.verification.pipeline.pipeline_evidence import EvidenceFlowInput
            fact_text = ctx.get_extra("prepared_fact") or ctx.get_extra("input_text") or ""
            stance_inp = EvidenceFlowInput(
                fact=fact_text,
                original_fact=fact_text,
                lang=ctx.lang or "en",
                content_lang=ctx.lang,
                analysis_mode=AnalysisMode.DEEP_V2,
                progress_callback=None,
            )
            
            total_pairs += len(chosen)
            result_items = await annotate_evidence_stance(
                agent=self.agent,
                inp=stance_inp,
                sources=chosen,
                claims=[claim],
            )

            # Ensure event_signature corresponds to the TARGET claim deterministically.
            md = claim.get("metadata") if isinstance(claim.get("metadata"), dict) else {}
            ents = claim.get("subject_entities") if isinstance(claim.get("subject_entities"), list) else []
            ts = md.get("time_signals") if isinstance(md.get("time_signals"), dict) else {}
            ls = md.get("locale_signals") if isinstance(md.get("locale_signals"), dict) else {}
            target_sig = {
                "entities": [str(x).strip()[:48] for x in ents[:5] if x],
                "time_bucket": str(ts.get("time_bucket") or ts.get("year") or "").strip()[:32],
                "locale": str(ls.get("country") or ls.get("locale") or "").strip()[:32],
            }

            # result_items are EvidenceItem-like dicts (same shape as sources)
            # Key by (claim_id, normalized_url) for safe replacement
            for ri in result_items or []:
                if not isinstance(ri, dict):
                    continue
                ri["event_signature"] = target_sig
                ru = ri.get("url")
                if not ru:
                    continue
                updated_map[(cid, normalize_url(str(ru)))] = ri

        if not updated_map:
            Trace.event("transferred_stance.completed", {"pairs": total_pairs, "updated": 0})
            return ctx

        # Merge updates into ctx.sources (replace matching transferred items)
        merged_sources: list[dict[str, Any]] = []
        updated = 0

        for s in sources:
            if not isinstance(s, dict):
                continue
            if s.get("provenance") != "transferred":
                merged_sources.append(s)
                continue

            cid = str(s.get("claim_id") or "")
            url = s.get("url")
            if not cid or not url:
                merged_sources.append(s)
                continue

            key = (cid, normalize_url(str(url)))
            repl = updated_map.get(key)
            if repl:
                # Preserve provenance/origin markers
                rr = dict(repl)
                rr["provenance"] = "transferred"
                rr["origin_claim_id"] = s.get("origin_claim_id")
                merged_sources.append(rr)
                updated += 1
            else:
                merged_sources.append(s)

        Trace.event(
            "transferred_stance.completed",
            {"pairs": total_pairs, "updated": updated, "top_k": top_k},
        )

        return ctx.with_update(sources=merged_sources).set_extra("transferred_stance_updated", updated)
