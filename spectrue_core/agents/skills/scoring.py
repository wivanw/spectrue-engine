# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.verification.evidence.evidence_pack import EvidencePack
from .base_skill import BaseSkill, logger
from spectrue_core.utils.trace import Trace
from spectrue_core.constants import SUPPORTED_LANGUAGES
from spectrue_core.utils.security import sanitize_input
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.agents.llm_client import is_schema_failure
from datetime import datetime
import hashlib
import asyncio
import json
from typing import Literal

from spectrue_core.llm.model_registry import ModelID
from spectrue_core.verification.scoring.judge_model_routing import (
    select_judge_model,
)

from spectrue_core.pipeline.mode import AnalysisMode

# Structured output schemas
from spectrue_core.agents.llm_schemas import (
    ANALYSIS_RESPONSE_SCHEMA,
    SCORE_EVIDENCE_STRUCTURED_SCHEMA,
    SCORING_RESPONSE_SCHEMA,
    SINGLE_CLAIM_SCORING_SCHEMA,
)

# Decomposition helpers
from .scoring_contract import (
    build_score_evidence_instructions,
    build_score_evidence_prompt,
    build_score_evidence_structured_instructions,
    build_score_evidence_structured_prompt,
    build_single_claim_scoring_instructions,
    build_single_claim_scoring_prompt,
)
from .scoring_parsing import clamp_score_evidence_result, parse_structured_verdict
from .scoring_sanitization import (
    MAX_QUOTE_LEN,
    MAX_SNIPPET_LEN,
    format_highlighted_excerpt,
    maybe_drop_style_section,
    sanitize_quote,
    sanitize_snippet,
    strip_internal_source_markers,
)

# Schema imports for structured scoring
from spectrue_core.schema import (
    ClaimUnit,
    StructuredVerdict,
    StructuredDebug,
)


class ScoringSkill(BaseSkill):

    async def score_evidence(
        self,
        pack: EvidencePack,
        *,
        lang: str = "en",
    ) -> dict:
        """
        Produce Final Verdict using Evidence Pack (Native Scoring v1.5 / M73.2 Platinum).
        Features: LLM-based aggregation, Deep Sanitization, Type Safety, UX Guardrails.
        ZERO-CRASH GUARANTEE: Robust handling of malformed lists/items.
        """
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")

        # --- 1. PREPARE EVIDENCE (Sanitized & Highlighted) ---
        sources_by_claim = {}

        # Prefer scored sources to avoid context-only hallucinations
        raw_results = pack.get("scored_sources")
        if not isinstance(raw_results, list) or len(raw_results) == 0:
            # If stance labeling produced no scored sources, still pass all retrieved sources.
            # Separation between decisive/context is preserved in EvidencePack; here we need
            # the judge to have access to non-homogeneous evidence rather than a hard empty.
            raw_results = pack.get("search_results")
        if not isinstance(raw_results, list):
            raw_results = []

        for r in raw_results:
            # PLATINUM FIX: Skip non-dict items to prevent crash on .get()
            if not isinstance(r, dict):
                continue

            cid = r.get("claim_id", "unknown")
            if cid not in sources_by_claim:
                sources_by_claim[cid] = []

            # SECURITY P0: Sanitize external content
            raw_snippet = r.get("content_excerpt") or r.get("snippet") or ""
            safe_snippet = sanitize_snippet(raw_snippet, limit=MAX_SNIPPET_LEN)

            key_snippet = (
                r.get("quote_span")
                or r.get("contradiction_span")
                or r.get("key_snippet")
            )
            stance = r.get("stance", "")

            # Visual cue: Highlight quotes found by Clustering
            content_text = format_highlighted_excerpt(
                safe_snippet=safe_snippet,
                key_snippet=key_snippet,
                stance=stance,
            )

            # FIX: Explicitly label PRIMARY sources for LLM
            if r.get("is_primary"):
                content_text = f"⭐ [PRIMARY SOURCE / OFFICIAL]\n{content_text}"

            sources_by_claim[cid].append({
                "domain": r.get("domain"),
                "link": r.get("url"),
                "title": r.get("title"),
                "source_reliability_hint": "high" if r.get("is_trusted") else "general",
                "stance": stance, 
                "excerpt": content_text
            })

        Trace.event(
            "score_evidence.inputs",
            {
                "claims": [
                    {
                        "claim_id": cid,
                        "sources": len(sources),
                        "stances": [
                            s.get("stance") for s in sources[:5]
                        ],
                    }
                    for cid, sources in sources_by_claim.items()
                ],
                "claims_total": len(sources_by_claim),
            },
        )

        # --- 2. PREPARE CLAIMS (Sanitized + Importance) ---
        claims_info = []
        raw_claims = pack.get("claims")
        if not isinstance(raw_claims, list):
            raw_claims = []

        for c in raw_claims:
            # PLATINUM FIX: Skip non-dict items
            if not isinstance(c, dict):
                continue

            cid = c.get("id")
            # SECURITY: Sanitize claims text
            safe_text = sanitize_input(c.get("text", ""))

            claims_info.append({
                "id": cid,
                "text": safe_text,
                "importance": float(c.get("importance", 0.5)), # Vital for LLM aggregation
                "matched_evidence_count": len(sources_by_claim.get(cid, []))
            })

        deferred_map = {
            c.get("id"): bool(c.get("deferred_from_search"))
            for c in raw_claims
            if isinstance(c, dict)
        }
        no_evidence = [
            c.get("id") for c in claims_info if c.get("matched_evidence_count", 0) == 0
        ]
        Trace.event(
            "score_evidence.coverage",
            {
                "claims_total": len(claims_info),
                "claims_with_sources": len(claims_info) - len(no_evidence),
                "claims_no_sources": no_evidence,
                "deferred_no_sources": [cid for cid in no_evidence if deferred_map.get(cid)],
            },
        )

        # SECURITY: Sanitize original fact
        safe_original_fact = sanitize_input(pack.get("original_fact", ""))

        instructions = build_score_evidence_instructions(lang_name=lang_name, lang=lang)

        prompt = build_score_evidence_prompt(
            safe_original_fact=safe_original_fact,
            claims_info=claims_info,
            sources_by_claim=sources_by_claim,
        )

        # Stable Cache Key
        prompt_hash = hashlib.sha256((instructions + prompt).encode()).hexdigest()[:32]
        cache_key = f"score_v7_plat_{prompt_hash}"

        try:
            m = self.config.openai_model or ModelID.PRO

            result = await self.llm_client.call_json(
                model=m,
                input=prompt,
                instructions=instructions,
                response_schema=SCORING_RESPONSE_SCHEMA,
                reasoning_effort="medium",
                cache_key=cache_key,
                timeout=float(self.runtime.llm.timeout_sec),
                trace_kind="score_evidence",
            )
            # Parse and normalize result
            clamped = clamp_score_evidence_result(result)

            # --- POST-PROCESSING: Force -1.0 for claims without evidence ---
            # LLM might hallucinate a score even with no evidence, or follow old habits.
            # We strictly enforce "Unverified" (-1.0) if matched_evidence_count == 0.
            claim_verdicts = clamped.get("claim_verdicts", [])
            claims_without_evidence = set(no_evidence)
            
            # Map for re-attaching text
            claim_text_map = {c["id"]: c["text"] for c in claims_info}

            for cv in claim_verdicts:
                cid = cv.get("claim_id")
                # UI requires 'text' to render the claim card
                if cid in claim_text_map:
                    cv["text"] = claim_text_map[cid]

                if cid in claims_without_evidence:
                    cv["verdict_score"] = -1.0
                    cv["verdict"] = "unverified"
                    cv["verdict_state"] = "insufficient_evidence"
                    cv["reason"] = cv.get("reason", "No evidence found for this claim.")
                    
                    # Update RGBA if present
                    if cv.get("rgba"):
                        # Keep R, B, A but set G to -1.0
                        # Or should we zero out A (explainability)?
                        r, g, b, a = cv["rgba"]
                        cv["rgba"] = [r, -1.0, b, 0.0] # Explainability 0 implies no evidence
            
            # --- POST-PROCESSING: Aggregate Global Scores ---
            # Deterministic aggregation MUST account for claims without evidence.
            # Contract: missing evidence is not "ignored"; it reduces (or blocks) the global score.
            # Policy:
            # - If the anchor/main claim is unverified -> global verified_score = -1.0
            # - Else compute importance-weighted mean where unverified contributes 0.0

            # Build claim importance map
            importance_by_id = {c["id"]: float(c.get("importance", 0.5) or 0.5) for c in claims_info if c.get("id")}

            # Heuristic anchor: highest importance claim (stable even if anchors missing)
            anchor_id = None
            if importance_by_id:
                anchor_id = max(importance_by_id.items(), key=lambda kv: kv[1])[0]

            # Determine per-claim scores (ensure every claim has an entry)
            score_by_id = {cv.get("claim_id"): cv.get("verdict_score", -1.0) for cv in claim_verdicts if cv.get("claim_id")}
            for ci in claims_info:
                cid = ci.get("id")
                if not cid:
                    continue
                if cid not in score_by_id:
                    score_by_id[cid] = -1.0

            if anchor_id and score_by_id.get(anchor_id, -1.0) < 0:
                clamped["verified_score"] = -1.0
                clamped["verified_score_policy"] = {
                    "policy": "anchor_unverified_blocks_global",
                    "anchor_claim_id": anchor_id,
                }
            else:
                num = 0.0
                den = 0.0
                for cid, imp in importance_by_id.items():
                    w = max(0.0, min(1.0, float(imp)))
                    g = float(score_by_id.get(cid, -1.0))
                    # Unverified -> contribute 0.0, but still counted in denominator
                    contrib = g if g >= 0 else 0.0
                    num += w * contrib
                    den += w
                clamped["verified_score"] = (num / den) if den > 0 else -1.0
                clamped["verified_score_policy"] = {
                    "policy": "importance_weighted_with_unverified_zero",
                    "anchor_claim_id": anchor_id,
                }
            
            # For trace/debug only: list of non-negative per-claim G values
            valid_g = [
                float(g) for g in score_by_id.values()
                if isinstance(g, (int, float)) and float(g) >= 0
            ]
            
            # Debug: trace aggregation decision
            Trace.event(
                "score_evidence.aggregation",
                {
                    "claim_count": len(claim_verdicts),
                    "claims_total": len(claims_info),
                    "anchor_claim_id": anchor_id,
                    "aggregation_policy": clamped.get("verified_score_policy"),
                    "valid_g_count": len(valid_g),
                    "valid_g_values": valid_g[:5] if valid_g else [],
                    "computed_verified_score": clamped["verified_score"],
                    "claims_without_evidence": list(claims_without_evidence)[:5],
                    "preserved_global_r": clamped.get("danger_score"),
                    "preserved_global_b": clamped.get("style_score"),
                    "preserved_global_a": clamped.get("explainability_score"),
                },
            )
            # NOTE: Do NOT aggregate R, B, A from claim-level RGBA
            # Those remain as global article-level scores from LLM

            metrics = pack.get("metrics", {})
            if isinstance(metrics, dict) and metrics.get("stance_failure") is True:
                current = clamped.get("explainability_score", -1.0)
                clamped["explainability_score"] = 0.2 if current < 0 else min(current, 0.4)
                clamped["stance_fallback"] = "context"

            return clamped

        except Exception as e:
            logger.exception("[Scoring] Failed: %s", e)
            Trace.event("llm.error", {"kind": "score_evidence", "error": str(e)})
            if is_schema_failure(e):
                raise
            return {
                "status": "error",
                "error": "llm_failed",
                "verified_score": -1.0,
                "explainability_score": -1.0,
                "danger_score": -1.0,
                "style_score": -1.0,
                "claim_verdicts": [],
                "rationale": "Error during analysis."
            }

    async def score_evidence_parallel(
        self,
        pack: EvidencePack,
        *,
        lang: str = "en",
        max_concurrency: int = 5,
    ) -> dict:
        """
        Score evidence with PARALLEL per-claim LLM calls.
        Each claim is scored independently, enabling true per-claim RGBA.
        
        Benefits:
        - No timeout on large prompts (each call is small)
        - True per-claim RGBA from LLM (not global scores copied)
        - Better scalability for many claims
        """
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")

        # --- 1. PREPARE EVIDENCE (Sanitized & Highlighted) ---
        sources_by_claim = {}

        raw_results = pack.get("scored_sources")
        if not isinstance(raw_results, list):
            raw_results = pack.get("search_results")
        if not isinstance(raw_results, list):
            raw_results = []

        for r in raw_results:
            if not isinstance(r, dict):
                continue

            cid = r.get("claim_id", "unknown")
            if cid not in sources_by_claim:
                sources_by_claim[cid] = []

            raw_snippet = r.get("content_excerpt") or r.get("snippet") or ""
            safe_snippet = sanitize_snippet(raw_snippet, limit=MAX_SNIPPET_LEN)

            key_snippet = (
                r.get("quote_span")
                or r.get("contradiction_span")
                or r.get("key_snippet")
            )
            stance = r.get("stance", "")

            content_text = format_highlighted_excerpt(
                safe_snippet=safe_snippet,
                key_snippet=key_snippet,
                stance=stance,
            )

            if r.get("is_primary"):
                content_text = f"⭐ [PRIMARY SOURCE / OFFICIAL]\n{content_text}"

            sources_by_claim[cid].append({
                "domain": r.get("domain"),
                "source_reliability_hint": "high" if r.get("is_trusted") else "general",
                "stance": stance, 
                "excerpt": content_text
            })

        # --- 2. PREPARE CLAIMS ---
        claims_info = []
        raw_claims = pack.get("claims")
        if not isinstance(raw_claims, list):
            raw_claims = []

        for c in raw_claims:
            if not isinstance(c, dict):
                continue

            cid = c.get("id")
            safe_text = sanitize_input(c.get("text", ""))

            claims_info.append({
                "id": cid,
                "text": safe_text,
                "importance": float(c.get("importance", 0.5)),
                "check_worthiness": float(c.get("check_worthiness", 0.5) or 0.5),
                "harm_potential": int(c.get("harm_potential", 1) or 1),
                "claim_role": c.get("claim_role") or c.get("role") or "unknown",
                "evidence_need": c.get("evidence_need") or "unknown",
                "claim_category": c.get("claim_category") or "FACTUAL",
                "metadata_confidence": c.get("metadata_confidence") or "unknown",
                "matched_evidence_count": len(sources_by_claim.get(cid, [])),
            })

        Trace.event(
            "score_evidence_parallel.start",
            {
                "claims_count": len(claims_info),
                "max_concurrency": max_concurrency,
            },
        )

        instructions = build_single_claim_scoring_instructions(lang_name=lang_name, lang=lang)

        # Helper wrapper to allow getattr access on dict keys
        class DictAttr:
            def __init__(self, d): self.__dict__ = d

        async def score_single_claim(claim_info: dict) -> dict:
            """Score a single claim with its evidence."""
            cid = claim_info.get("id")
            evidence = sources_by_claim.get(cid, [])

            prompt = build_single_claim_scoring_prompt(
                claim_info=claim_info,
                evidence=evidence,
            )

            prompt_hash = hashlib.sha256((instructions + prompt).encode()).hexdigest()[:32]
            cache_key = f"score_single_v1_{prompt_hash}"

            # --- MODEL ROUTING using spectrue_core.verification.scoring.judge_model_routing ---
            # Fast-path: no evidence => deterministic unverified; no LLM call.
            if not evidence:
                return {
                    "claim_id": cid,
                    "text": claim_info.get("text", ""),
                    "status": "ok",
                    "verdict_score": -1.0,
                    "verdict": "unverified",
                    "verdict_state": "insufficient_evidence",
                    "reason": "No evidence found for this claim.",
                    "rgba": [0.0, -1.0, 0.0, 0.0],
                    "routing": {"model": "__skip__", "reason": "no_evidence"},
                }

            routed_model: str = ModelID.PRO
            route_debug = {}
            
            try:
                # Wrap inputs for getattr-based routing
                c_obj = DictAttr(claim_info)
                ev_objs = [DictAttr(e) for e in evidence]
                
                decision = select_judge_model(
                    claim=c_obj,
                    evidence_items=ev_objs,
                    prompt_chars=len(prompt),
                    out_tokens_estimate=int(getattr(self.runtime.llm, "judge_out_tokens_estimate", 380)),
                    deepseek_fail_prob=float(getattr(self.runtime.llm, "deepseek_fail_prob", 0.18)),
                )
                routed_model = decision.model
                route_debug = decision.to_trace()
            except Exception as e:
                # Fallback to PRO on any routing error
                routed_model = ModelID.PRO
                route_debug = {
                    "model": ModelID.PRO,
                    "reason": f"routing_error: {str(e)}",
                    "features": {"error": str(e)}
                }

            Trace.event("judge.model_route", route_debug)
            
            # Use standard deepseek model name if routing returned 'deepseek-chat'
            deepseek_names = tuple(getattr(self.runtime.llm, "deepseek_model_names", ()) or ())
            actual_deepseek_model = deepseek_names[0] if deepseek_names else ModelID.MID
            if routed_model == ModelID.MID:
                 routed_model = actual_deepseek_model

            async def _call_model(model_name: str, *, use_cache: bool) -> dict:
                """Repair-retry protocol for a single model. Returns dict with status ok/error."""
                # Try up to 2 times: first normal call, then repair if invalid
                max_attempts = 2
                last_error: str | None = None

                local_prompt = prompt
                missing_fields: list[str] = []

                # Timeouts / effort
                if model_name == ModelID.NANO:
                    timeout_s = float(self.runtime.llm.nano_timeout_sec)
                    effort: Literal["low", "medium", "high"] = "low"
                elif model_name == ModelID.PRO:
                    timeout_s = float(self.runtime.llm.timeout_sec)
                    effort = "low"
                else:
                    timeout_s = float(getattr(self.runtime.llm, "cluster_timeout_sec", self.runtime.llm.timeout_sec))
                    effort = "low"

                for attempt in range(1, max_attempts + 1):
                    try:
                        result = await self.llm_client.call_json(
                            model=model_name,
                            input=local_prompt,
                            instructions=instructions,
                            response_schema=SINGLE_CLAIM_SCORING_SCHEMA,
                            reasoning_effort=effort,
                            cache_key=cache_key if (use_cache and attempt == 1) else None,
                            timeout=timeout_s,
                            trace_kind="score_single_claim",
                        )

                        # Validate required fields
                        missing_fields = []
                        if result.get("verdict_score") is None:
                            missing_fields.append("verdict_score")
                        if not result.get("verdict"):
                            missing_fields.append("verdict")

                        rgba = result.get("rgba")
                        rgba_valid = (
                            isinstance(rgba, list) 
                            and len(rgba) >= 4 
                            and all(isinstance(x, (int, float)) for x in rgba[:4])
                        )
                        if not rgba_valid:
                            missing_fields.append("rgba")

                        # Log schema mismatch details
                        if missing_fields:
                            Trace.event(
                                "score_single_claim.schema_mismatch",
                                {
                                    "claim_id": cid,
                                    "attempt": attempt,
                                    "model": model_name,
                                    "missing_fields": missing_fields,
                                    "received_keys": list(result.keys()),
                                    "verdict_score_raw": result.get("verdict_score"),
                                    "verdict_raw": result.get("verdict"),
                                    "rgba_raw": result.get("rgba"),
                                },
                            )

                        # If invalid and first attempt, try repair
                        if missing_fields and attempt < max_attempts:
                            Trace.event(
                                "score_single_claim.repair_needed",
                                {"claim_id": cid, "attempt": attempt, "model": model_name, "missing_fields": missing_fields},
                            )
                            # Modify prompt for repair
                            local_prompt = f"""Your previous response was missing required fields: {missing_fields}.
Please return a complete JSON with ALL required fields:
- claim_id (string)
- verdict_score (float 0.0-1.0)
- verdict (string: verified/refuted/ambiguous/unverified)
- reason (string)
- rgba (array of 4 floats [R,G,B,A])

Original claim:
{json.dumps(claim_info, indent=2, ensure_ascii=False)}

Return valid JSON now."""
                            continue

                        # If still missing fields after repair, mark as error
                        if missing_fields:
                            return {
                                "claim_id": cid,
                                "status": "error",
                                "error_type": "schema_invalid",
                                "missing_fields": missing_fields,
                                "verdict_score": None,
                                "verdict": None,
                                "rgba": None,
                                "reason": f"LLM response missing required fields: {missing_fields}",
                                "model": model_name,
                            }

                        # SUCCESS: All fields valid
                        result["claim_id"] = cid
                        result["text"] = claim_info.get("text", "")
                        result["status"] = "ok"
                        result["model"] = model_name

                        # Clamp RGBA values
                        # Allow -1.0 for G (verified_score) to signal "unknown"
                        g_val = float(rgba[1])
                        if g_val < 0:
                            g_val = -1.0
                        else:
                            g_val = max(0.0, min(1.0, g_val))

                        result["rgba"] = [
                            max(0.0, min(1.0, float(rgba[0]))),
                            g_val,                               # G (verified)
                            max(0.0, min(1.0, float(rgba[2]))),
                            max(0.0, min(1.0, float(rgba[3]))),
                        ]

                        Trace.event(
                            "score_single_claim.success",
                            {
                                "claim_id": cid,
                                "model": model_name,
                                "verdict_score": result.get("verdict_score"),
                                "rgba": result.get("rgba"),
                                "attempts": attempt,
                                "repair_used": attempt > 1,
                            },
                        )

                        return result

                    except Exception as e:
                        last_error = str(e)
                        Trace.event(
                            "score_single_claim.llm_error",
                            {
                                "claim_id": cid,
                                "attempt": attempt,
                                "model": model_name,
                                "error": last_error[:500],
                                "error_type": type(e).__name__,
                            },
                        )
                        if attempt < max_attempts:
                            Trace.event(
                                "score_single_claim.retry",
                                {"claim_id": cid, "attempt": attempt, "model": model_name, "error": last_error[:200]},
                            )
                            continue

                # All attempts failed
                Trace.event(
                    "score_single_claim.failed",
                    {"claim_id": cid, "model": model_name, "error": (last_error or "unknown")[:500], "attempts": max_attempts},
                )
                return {
                    "claim_id": cid,
                    "status": "error",
                    "error_type": "llm_failed",
                    "verdict_score": None,
                    "verdict": None,
                    "rgba": None,
                    "reason": f"LLM call failed: {last_error}",
                    "model": model_name,
                }

            # 1) Try routed model
            primary = await _call_model(routed_model, use_cache=True)
            if primary.get("status") == "ok":
                primary["routing"] = route_debug
                return primary

            # 2) DeepSeek is flaky: hard fallback to gpt-5.2 on ANY failure
            if routed_model == actual_deepseek_model:
                Trace.event(
                    "judge.model_fallback",
                    {
                        "claim_id": cid,
                        "from_model": routed_model,
                        "to_model": "gpt-5.2",
                        "reason": primary.get("error_type") or "unknown",
                    },
                )
                fallback = await _call_model("gpt-5.2", use_cache=False)
                if fallback.get("status") == "ok":
                    fallback["routing"] = {**route_debug, "fallback_from": routed_model}
                    return fallback
                fallback["routing"] = {**route_debug, "fallback_from": routed_model}
                fallback["fallback_failed"] = True
                return fallback

            # Optional: nano failure fallback (rare but safer than returning error)
            if routed_model == "gpt-5-nano":
                Trace.event(
                    "judge.model_fallback",
                    {
                        "claim_id": cid,
                        "from_model": routed_model,
                        "to_model": "gpt-5.2",
                        "reason": primary.get("error_type") or "unknown",
                    },
                )
                fallback = await _call_model("gpt-5.2", use_cache=False)
                if fallback.get("status") == "ok":
                    fallback["routing"] = {**route_debug, "fallback_from": routed_model}
                    return fallback
                fallback["routing"] = {**route_debug, "fallback_from": routed_model}
                fallback["fallback_failed"] = True
                return fallback

            primary["routing"] = route_debug
            return primary

        # Limit concurrency with semaphore
        semaphore = asyncio.Semaphore(max_concurrency)

        async def score_with_semaphore(claim_info: dict) -> dict:
            async with semaphore:
                return await score_single_claim(claim_info)

        # Run all claims in parallel
        tasks = [score_with_semaphore(ci) for ci in claims_info]
        claim_verdicts = await asyncio.gather(*tasks)

        for cv in claim_verdicts:
            if cv.get("status") != "ok":
                cv["rgba"] = None
                cv["verdict_score"] = None

        # --- 4. AGGREGATE GLOBAL SCORES ---
        # Calculate global scores from per-claim RGBA
        # IMPORTANT: Skip claims with status=error (they have rgba=None)
        valid_verdicts = [cv for cv in claim_verdicts if cv.get("status") == "ok" and cv.get("rgba")]
        error_verdicts = [cv for cv in claim_verdicts if cv.get("status") == "error"]

        if error_verdicts:
            Trace.event(
                "score_evidence_parallel.errors",
                {
                    "error_count": len(error_verdicts),
                    "error_claims": [cv.get("claim_id") for cv in error_verdicts],
                },
            )

        if valid_verdicts:
            all_r = [cv["rgba"][0] for cv in valid_verdicts]
            # Handle -1.0 in G (ignore them for average, unless ALL are -1)
            raw_g = [cv["rgba"][1] for cv in valid_verdicts]
            valid_g_values = [g for g in raw_g if g >= 0]
            
            all_b = [cv["rgba"][2] for cv in valid_verdicts]
            all_a = [cv["rgba"][3] for cv in valid_verdicts]

            global_danger = sum(all_r) / len(all_r)
            
            if valid_g_values:
                global_verified = sum(valid_g_values) / len(valid_g_values)
            else:
                # All claims are unknown/unverified
                global_verified = -1.0 # Signal unknown
                
            global_style = sum(all_b) / len(all_b)
            global_explain = sum(all_a) / len(all_a)
        else:
            # All claims failed — cannot compute scores
            global_danger = None
            global_verified = None
            global_style = None
            global_explain = None

        # Build rationale from claim reasons (only valid verdicts)
        rationale_parts = []
        for cv in valid_verdicts:
            reason = cv.get("reason", "")
            if reason:
                rationale_parts.append(reason)

        global_rationale = " ".join(rationale_parts[:3])  # First 3 reasons
        if not global_rationale:
            if error_verdicts:
                global_rationale = f"Analysis incomplete: {len(error_verdicts)} claim(s) failed to score."
            else:
                global_rationale = "Analysis completed."

        result = {
            "claim_verdicts": list(claim_verdicts),
            "verified_score": global_verified,
            "explainability_score": global_explain,
            "danger_score": global_danger,
            "style_score": global_style,
            "rationale": global_rationale,
            # Track scoring quality
            "scoring_stats": {
                "total_claims": len(claim_verdicts),
                "valid_claims": len(valid_verdicts),
                "error_claims": len(error_verdicts),
            },
        }

        Trace.event(
            "score_evidence_parallel.completed",
            {
                "claims_scored": len(claim_verdicts),
                "global_verified": global_verified,
                "global_danger": global_danger,
            },
        )

        return result


    def detect_evidence_gaps(self, pack: EvidencePack) -> list[str]:
        """
        Identify missing evidence types (Legacy/Informational).
        """
        gaps = []

        # Harden: ensure metrics and per_claim are dicts
        metrics = pack.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        per_claim = metrics.get("per_claim", {})
        if not isinstance(per_claim, dict):
            per_claim = {}

        # Harden: ensure claims is a list
        claims = pack.get("claims")
        if not isinstance(claims, list):
            claims = []

        for cid, m in per_claim.items():
            # Skip non-dict entries
            if not isinstance(m, dict):
                continue

            if m.get("independent_domains", 0) < 2:
                gaps.append(f"Claim {cid}: Low source diversity (informational).")

            claim_obj = next(
                (c for c in claims if isinstance(c, dict) and c.get("id") == cid),
                None
            )
            if claim_obj:
                req = claim_obj.get("evidence_requirement", {})
                if req.get("needs_primary_source") and not m.get("primary_present"):
                     gaps.append(f"Claim {cid}: Missing primary source (informational).")

        return gaps

    def _strip_internal_source_markers(self, text: str) -> str:
        return strip_internal_source_markers(text)

    def _maybe_drop_style_section(self, rationale: str, *, honesty_score: float | None, lang: str | None) -> str:
        return maybe_drop_style_section(rationale, honesty_score=honesty_score, lang=lang)

    async def analyze(self, fact: str, context: str, lang: str, analysis_mode: str | AnalysisMode = AnalysisMode.GENERAL) -> dict:
        """Analyze fact based on context and return structured JSON."""
        
        prompt_key = 'prompts.final_analysis'
        prompt_template = get_prompt(lang, prompt_key)

        safe_fact = f"<statement>{sanitize_input(fact)}</statement>"
        safe_context = f"<context>{sanitize_input(context)}</context>"

        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = prompt_template.format(fact=safe_fact, context=safe_context, date=current_date)

        no_tag_note_by_lang = {
            "uk": "ВАЖЛИВО: Не згадуй у відповіді службові мітки джерел на кшталт [TRUSTED], [REL=…], [RAW].",
            "en": "IMPORTANT: Do not mention internal source tags like [TRUSTED], [REL=…], [RAW] in your output.",
            "ru": "ВАЖНО: Не упоминай в ответе служебные метки источников вроде [TRUSTED], [REL=…], [RAW].",
            "de": "WICHTIG: Erwähne keine internen Quellenmarkierungen wie [TRUSTED], [REL=…], [RAW] in deiner Antwort.",
            "es": "IMPORTANTE: No menciones etiquetas internas de fuentes como [TRUSTED], [REL=…], [RAW] en tu respuesta.",
            "fr": "IMPORTANT: Ne mentionne pas les balises internes de sources comme [TRUSTED], [REL=…], [RAW] dans ta réponse.",
            "ja": "重要: 出力に[TRUSTED]、[REL=…]、[RAW]などの内部ソースタグを含めないでください。",
            "zh": "重要：请勿在输出中提及内部来源标签，如 [TRUSTED]、[REL=…]、[RAW]。",
        }
        prompt += "\n\n" + no_tag_note_by_lang.get((lang or "").lower(), no_tag_note_by_lang["en"])

        Trace.event("llm.prompt", {"kind": "analysis", "model": self.config.openai_model})
        try:
            result = await self.llm_client.call_json(
                model=self.config.openai_model or "gpt-5-nano",
                input=prompt,
                instructions="You are Aletheia, an advanced fact-checking AI.",
                response_schema=ANALYSIS_RESPONSE_SCHEMA,
                reasoning_effort="low",
                cache_key=f"final_analysis_{lang}_v1",
                timeout=float(self.runtime.llm.timeout_sec),
                trace_kind="analysis",
            )
            Trace.event("llm.parsed", {"kind": "analysis", "keys": list(result.keys())})

        except Exception as e:
            logger.exception("[GPT] ✗ Error calling %s: %s", self.llm_client.model, e)
            Trace.event("llm.error", {"kind": "analysis", "error": str(e)})
            if is_schema_failure(e):
                raise
            return {
                "verified_score": -1.0, "context_score": -1.0, "danger_score": -1.0,
                "style_score": -1.0, "confidence_score": -1.0, "rationale_key": "errors.agent_call_failed"
            }

        if "rationale" in result:
             # Clean up markers and style section
             cleaned_rationale = self._strip_internal_source_markers(str(result.get("rationale") or ""))

             # Calculate legacy honesty score for style dropping logic
             try:
                cs = float(result.get("context_score", 0.5))
                ss = float(result.get("style_score", 0.5))
                honesty = (max(0.0, min(1.0, cs)) + max(0.0, min(1.0, ss))) / 2.0
             except Exception:
                honesty = None

             result["rationale"] = self._maybe_drop_style_section(cleaned_rationale, honesty_score=honesty, lang=lang)

        if "analysis" in result and isinstance(result["analysis"], str):
             result["analysis"] = self._strip_internal_source_markers(result["analysis"])

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Schema-First Scoring (Per-Assertion Verdicts)
    # ─────────────────────────────────────────────────────────────────────────

    async def score_evidence_structured(
        self,
        claim_units: list[ClaimUnit],
        evidence: list[dict],
        *,
        lang: str = "en",
    ) -> StructuredVerdict:
        """
        Score claims with per-assertion verdicts.
        
        Key Design:
        - FACT assertions: Strictly verified (can be VERIFIED/REFUTED/AMBIGUOUS)
        - CONTEXT assertions: Soft verification (only VERIFIED/AMBIGUOUS, rarely REFUTED)
        - CRITICAL: CONTEXT evidence cannot refute FACT assertions
        - Scores use -1.0 sentinel for "not computed"
        
        Args:
            claim_units: List of structured ClaimUnit objects
            evidence: List of evidence dicts with assertion_key
            model: LLM model to use
            lang: Output language for rationale
            
        Returns:
            StructuredVerdict with per-claim and per-assertion verdicts
        """
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")

        if not claim_units:
            return StructuredVerdict(
                verified_score=-1.0,
                explainability_score=-1.0,
                danger_score=-1.0,
                style_score=-1.0,
                rationale="No claims to verify.",
            )

        # --- 1. PREPARE CLAIMS WITH ASSERTIONS ---
        claims_data = []
        for unit in claim_units:
            assertions_data = []
            for a in unit.assertions:
                assertions_data.append({
                    "key": a.key,
                    "value": str(a.value) if a.value else "",
                    "dimension": a.dimension.value,  # FACT / CONTEXT / INTERPRETATION
                    "importance": a.importance,
                    "is_inferred": a.is_inferred,
                })

            claims_data.append({
                "id": unit.id,
                "text": sanitize_input(unit.text or unit.normalized_text),
                "type": unit.claim_type.value,
                "importance": unit.importance,
                "assertions": assertions_data,
            })

        # --- 2. PREPARE EVIDENCE BY ASSERTION ---
        evidence_by_assertion: dict[str, list[dict]] = {}
        for e in evidence:
            if not isinstance(e, dict):
                continue

            claim_id = e.get("claim_id", "")
            assertion_key = e.get("assertion_key", "")
            map_key = f"{claim_id}:{assertion_key}" if assertion_key else claim_id

            if map_key not in evidence_by_assertion:
                evidence_by_assertion[map_key] = []

            raw_content = e.get("content_excerpt") or e.get("snippet") or ""
            safe_content = sanitize_snippet(raw_content, limit=MAX_SNIPPET_LEN)

            key_snippet = (
                e.get("quote_span")
                or e.get("contradiction_span")
                or e.get("key_snippet")
                or e.get("quote")
            )
            stance = e.get("stance", "MENTION")

            if key_snippet and stance in ["SUPPORT", "REFUTE", "MIXED"]:
                safe_quote = sanitize_quote(key_snippet, limit=MAX_QUOTE_LEN)
                content_text = f'📌 QUOTE: "{safe_quote}"\nℹ️ CONTEXT: {safe_content}'
            else:
                content_text = safe_content

            # FIX: Explicitly label PRIMARY sources for LLM
            if e.get("is_primary"):
                content_text = f"⭐ [PRIMARY SOURCE / OFFICIAL]\n{content_text}"

            evidence_by_assertion[map_key].append({
                "domain": e.get("domain"),
                "stance": stance,
                "content_status": e.get("content_status", "available"),
                "excerpt": content_text,
                "is_trusted": e.get("is_trusted", False),
            })

        instructions = build_score_evidence_structured_instructions(lang_name=lang_name, lang=lang)

        prompt = build_score_evidence_structured_prompt(
            claims_data=claims_data,
            evidence_by_assertion=evidence_by_assertion,
        )

        # Cache key
        prompt_hash = hashlib.sha256((instructions + prompt).encode()).hexdigest()[:32]
        cache_key = f"score_struct_v1_{prompt_hash}"

        try:
            m = self.config.openai_model or "gpt-5.2"

            result = await self.llm_client.call_json(
                model=m,
                input=prompt,
                instructions=instructions,
                response_schema=SCORE_EVIDENCE_STRUCTURED_SCHEMA,
                reasoning_effort="medium",
                cache_key=cache_key,
                timeout=float(self.runtime.llm.timeout_sec),
                trace_kind="score_evidence_structured",
            )

            # --- 5. PARSE LLM RESPONSE INTO StructuredVerdict ---
            return self._parse_structured_verdict(result, lang=lang)

        except Exception as e:
            logger.exception("[Scoring] Failed: %s", e)
            Trace.event("llm.error", {"kind": "score_evidence_structured", "error": str(e)})
            if is_schema_failure(e):
                raise
            return StructuredVerdict(
                verified_score=-1.0,
                explainability_score=-1.0,
                danger_score=-1.0,
                style_score=-1.0,
                rationale="Error during analysis.",
                structured_debug=StructuredDebug(
                    processing_notes=[f"LLM error: {str(e)}"]
                ),
            )

    def _parse_structured_verdict(self, raw: dict, *, lang: str = "en") -> StructuredVerdict:
        """
        Parse LLM response into StructuredVerdict.
        
        Uses -1.0 sentinel for missing scores.
        """
        return parse_structured_verdict(raw, lang=lang)
