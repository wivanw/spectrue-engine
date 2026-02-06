"""Evidence flow orchestration logic."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from spectrue_core.domain.verification.verdict.model import (
    AnalysisMode,
    EvidenceFlowInput,
    EvidenceCollection,
    SearchProfileName,
    resolve_stance_pass_mode,
)
from spectrue_core.utils.source_utils import canonicalize_sources
from spectrue_core.utils.temporal import (
    label_evidence_timeliness,
    normalize_time_window,
)
from spectrue_core.utils.claim_selection import pick_ui_main_claim
from spectrue_core.utils.evidence import build_evidence_pack
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)

async def annotate_evidence_stance(
    *,
    agent,
    inp: EvidenceFlowInput,
    claims: list[dict],
    sources: list[dict],
) -> list[dict]:
    """Optional stance annotation using clustering skill output."""
    if not claims or not sources:
        return []
    if inp.progress_callback:
        await inp.progress_callback("stance_annotation", 0.0, None)
    
    # Map analysis_mode to profile
    if inp.analysis_mode in (AnalysisMode.DEEP, AnalysisMode.DEEP_V2):
        profile_name = SearchProfileName.DEEP
    else:
        profile_name = SearchProfileName.GENERAL

    stance_pass_mode = resolve_stance_pass_mode(profile_name)
    evidence_items = await agent.cluster_evidence(
        claims,
        sources,
        stance_pass_mode=stance_pass_mode,
    )

    # Mapping of claim components for normalization
    KNOWN_COVERS = {"entity", "time", "location", "quantity", "attribution", "causal", "other"}

    for ev in evidence_items or []:
        if not isinstance(ev, dict):
            continue

        raw_covers = ev.get("covers")
        normalized_covers = []
        if isinstance(raw_covers, list):
            for c in raw_covers:
                c_norm = str(c).strip().lower()
                if c_norm in KNOWN_COVERS:
                    normalized_covers.append(c_norm)

        ev["covers"] = list(set(normalized_covers))

    # Deterministic event signature stamping
    claim_lookup = {}
    for idx, c in enumerate(claims or []):
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or c.get("claim_id") or f"c{idx+1}")
        claim_lookup[cid] = c

    for ev in evidence_items or []:
        if not isinstance(ev, dict):
            continue
        cid = ev.get("claim_id")
        if not cid:
            continue
        claim = claim_lookup.get(str(cid))
        if not isinstance(claim, dict):
            continue

        md = claim.get("metadata") if isinstance(claim.get("metadata"), dict) else {}
        ents = claim.get("subject_entities") if isinstance(claim.get("subject_entities"), list) else []
        ts = md.get("time_signals") if isinstance(md.get("time_signals"), dict) else {}
        ls = md.get("locale_signals") if isinstance(md.get("locale_signals"), dict) else {}

        ev["event_signature"] = {
            "entities": [str(x).strip()[:48] for x in ents[:5] if x],
            "time_bucket": str(ts.get("time_bucket") or ts.get("year") or "").strip()[:32],
            "locale": str(ls.get("country") or ls.get("locale") or "").strip()[:32],
        }

    return evidence_items


async def collect_evidence(
    *,
    agent,
    search_mgr,
    inp: EvidenceFlowInput,
    claims: list[dict],
    sources: list[dict],
    calibration_registry = None,
) -> EvidenceCollection:
    """Collect and structure evidence without invoking the judge."""
    if inp.progress_callback:
        await inp.progress_callback("ai_analysis", 0.0, None)

    current_cost = search_mgr.calculate_cost()
    sources = canonicalize_sources(sources)

    time_windows: dict[str, Any] = {}
    if claims:
        default_relative_days = getattr(
            getattr(getattr(search_mgr, "config", None), "runtime", None),
            "temporal",
            None,
        )
        default_days = getattr(default_relative_days, "relative_window_days", None)
        default_days = int(default_days) if isinstance(default_days, int) else None

        for claim in claims:
            claim_id = str(claim.get("id") or "c1")
            metadata = claim.get("metadata")
            time_signals = []
            time_sensitive = False
            if metadata:
                time_signals = list(getattr(metadata, "time_signals", []) or [])
                time_sensitive = bool(getattr(metadata, "time_sensitive", False))

            req = claim.get("evidence_requirement") or {}
            if isinstance(req, dict) and req.get("is_time_sensitive"):
                time_sensitive = True
            if isinstance(req, dict) and req.get("needs_recent_source"):
                time_sensitive = True

            time_windows[claim_id] = normalize_time_window(
                time_signals if (time_signals or time_sensitive) else [],
                reference_date=date.today(),
                default_relative_days=default_days or 30,
            )

        for claim_id, window in time_windows.items():
            claim_sources = [s for s in sources if s.get("claim_id") == claim_id]
            if not claim_sources and len(time_windows) == 1:
                label_evidence_timeliness(sources, time_window=window)
            else:
                label_evidence_timeliness(claim_sources, time_window=window)

    claim_text_map: dict[str, str] = {}
    if claims:
        for c in claims:
            if not isinstance(c, dict):
                continue
            cid = c.get("id") or c.get("claim_id")
            if not cid:
                continue
            claim_text_map[str(cid)] = c.get("normalized_text") or c.get("text") or ""

    anchor_claim = None
    anchor_claim_id = None
    if claims:
        anchor_claim = pick_ui_main_claim(claims, calibration_registry=calibration_registry) or claims[0]
        anchor_claim_id = anchor_claim.get("id") or anchor_claim.get("claim_id")

    # Language consistency validation
    if claims and inp.content_lang:
        from spectrue_core.utils.language_validation import validate_claims_language_consistency
        lang_valid, lang_mismatches = validate_claims_language_consistency(
            claims, inp.content_lang, pipeline_mode="collect", min_confidence=0.7,
        )
        if not lang_valid:
            Trace.event("pipeline.language_mismatch_ignored", {
                "expected": inp.content_lang, "mismatches": lang_mismatches,
            })

    pack = build_evidence_pack(
        fact=inp.original_fact,
        claims=claims,
        sources=sources,
        search_results_clustered=None,
        content_lang=inp.content_lang or inp.lang,
        article_context={"text_excerpt": inp.fact[:500]} if inp.fact != inp.original_fact else None,
        anchor_claim_id=anchor_claim_id,
    )

    return EvidenceCollection(
        pack=pack,
        claims=claims,
        sources=sources,
        claim_text_map=claim_text_map,
        anchor_claim=anchor_claim,
        anchor_claim_id=anchor_claim_id,
        time_windows=time_windows,
        current_cost=current_cost,
    )
