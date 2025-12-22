from spectrue_core.verification.evidence_pack import EvidencePack
from .base_skill import BaseSkill, logger
from spectrue_core.utils.trace import Trace
from spectrue_core.constants import SUPPORTED_LANGUAGES
from spectrue_core.utils.security import sanitize_input
from spectrue_core.agents.prompts import get_prompt
from datetime import datetime
import hashlib

# M87: Decomposition helpers
from .scoring_contract import (
    build_score_evidence_instructions,
    build_score_evidence_prompt,
    build_score_evidence_structured_instructions,
    build_score_evidence_structured_prompt,
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

# M70: Schema imports for structured scoring
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
        model: str = "gpt-5.2",
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
        
        # Defensive: ensure search_results is a list
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
            
            key_snippet = r.get("key_snippet")
            stance = r.get("stance", "")
            
            # Visual cue: Highlight quotes found by Clustering
            content_text = format_highlighted_excerpt(
                safe_snippet=safe_snippet,
                key_snippet=key_snippet,
                stance=stance,
            )
            
            sources_by_claim[cid].append({
                "domain": r.get("domain"),
                "source_reliability_hint": "high" if r.get("is_trusted") else "general",
                "stance": stance, 
                "excerpt": content_text
            })
            
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
            m = model if "gpt-5" in model or "o1" in model else "gpt-5.2"

            result = await self.llm_client.call_json(
                model=m,
                input=prompt,
                instructions=instructions,
                reasoning_effort="medium",
                cache_key=cache_key,
                timeout=float(self.runtime.llm.timeout_sec),
                trace_kind="score_evidence",
            )
            return clamp_score_evidence_result(result)

        except Exception as e:
            logger.exception("[Scoring] Failed: %s", e)
            Trace.event("llm.error", {"kind": "score_evidence", "error": str(e)})
            return {
                "verified_score": -1.0,
                "explainability_score": -1.0,
                "danger_score": -1.0,
                "style_score": -1.0,
                "rationale": "Error during analysis."
            }


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

    async def analyze(self, fact: str, context: str, gpt_model: str, lang: str, analysis_mode: str = "general") -> dict:
        """Analyze fact based on context and return structured JSON."""
        
        prompt_key = 'prompts.aletheia_lite' if analysis_mode == 'lite' else 'prompts.final_analysis'
        prompt_template = get_prompt(lang, prompt_key)
        
        if not prompt_template and analysis_mode == 'lite':
             prompt_template = get_prompt(lang, 'prompts.final_analysis')

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
        
        Trace.event("llm.prompt", {"kind": "analysis", "model": gpt_model})

        result = {}
        try:
            result = await self.llm_client.call_json(
                model=gpt_model,
                input=prompt,
                instructions="You are Aletheia, an advanced fact-checking AI.",
                reasoning_effort="low",
                cache_key=f"final_analysis_{lang}_v1",
                timeout=float(self.runtime.llm.timeout_sec),
                trace_kind="analysis",
            )
            Trace.event("llm.parsed", {"kind": "analysis", "keys": list(result.keys())})

        except Exception as e:
            logger.exception("[GPT] ✗ Error calling %s: %s", gpt_model, e)
            Trace.event("llm.error", {"kind": "analysis", "error": str(e)})
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
    # M70: Schema-First Scoring (Per-Assertion Verdicts)
    # ─────────────────────────────────────────────────────────────────────────

    async def score_evidence_structured(
        self,
        claim_units: list[ClaimUnit],
        evidence: list[dict],
        *,
        model: str = "gpt-5.2",
        lang: str = "en",
    ) -> StructuredVerdict:
        """
        M70: Score claims with per-assertion verdicts.
        
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
            
            key_snippet = e.get("key_snippet") or e.get("quote")
            stance = e.get("stance", "MENTION")
            
            if key_snippet and stance in ["SUPPORT", "REFUTE", "MIXED"]:
                safe_quote = sanitize_quote(key_snippet, limit=MAX_QUOTE_LEN)
                content_text = f'📌 QUOTE: "{safe_quote}"\nℹ️ CONTEXT: {safe_content}'
            else:
                content_text = safe_content
            
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
            m = model if "gpt-5" in model or "o1" in model else "gpt-5.2"

            result = await self.llm_client.call_json(
                model=m,
                input=prompt,
                instructions=instructions,
                reasoning_effort="medium",
                cache_key=cache_key,
                timeout=float(self.runtime.llm.timeout_sec),
                trace_kind="score_evidence_structured",
            )

            # --- 5. PARSE LLM RESPONSE INTO StructuredVerdict ---
            return self._parse_structured_verdict(result, lang=lang)

        except Exception as e:
            logger.exception("[M70 Scoring] Failed: %s", e)
            Trace.event("llm.error", {"kind": "score_evidence_structured", "error": str(e)})
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
        M70: Parse LLM response into StructuredVerdict.
        
        Uses -1.0 sentinel for missing scores.
        """
        return parse_structured_verdict(raw, lang=lang)
