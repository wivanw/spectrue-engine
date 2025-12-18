from spectrue_core.verification.evidence_pack import EvidencePack
from .base_skill import BaseSkill, logger
from spectrue_core.utils.trace import Trace
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.agents.static_instructions import UNIVERSAL_METHODOLOGY_APPENDIX
from spectrue_core.utils.security import sanitize_input
from spectrue_core.constants import SUPPORTED_LANGUAGES
from datetime import datetime
import re

class ScoringSkill(BaseSkill):

    async def score_evidence(
        self,
        pack: EvidencePack,
        *,
        model: str = "gpt-5.2",
        lang: str = "en",
    ) -> dict:
        """
        Produce Final Verdict using Evidence Pack (M48).
        """
        # M57: Resolve language name for explicit rationale localization
        lang_name = SUPPORTED_LANGUAGES.get(lang.lower(), "English")
        
        # Group search results by claim for the prompt
        sources_by_claim = {}
        for r in (pack.get("search_results") or []):
            cid = r.get("claim_id", "unknown")
            if cid not in sources_by_claim:
                sources_by_claim[cid] = []
            sources_by_claim[cid].append({
                "domain": r.get("domain"),
                "source_type": r.get("source_type"),
                "stance": r.get("stance"),
                "is_trusted": r.get("is_trusted"),
                "relevance": r.get("relevance_score"),
                "excerpt": (r.get("content_excerpt") or "")[:500]
            })
            
        constraints = pack.get("constraints") or {}
        global_cap = constraints.get("global_cap", 1.0)
        cap_reasons = constraints.get("cap_reasons", [])

        # Construct prompt
        # Prepare detailed claim info for the prompt
        claims_info = []
        for c in (pack.get("claims") or []):
            cid = c.get("id")
            # Get evidence specific to this claim (if mapped) or all relevant evidence
            claim_evidence = sources_by_claim.get(cid, [])
            
            claims_info.append({
                "id": cid,
                "text": c.get("text"),
                "type": c.get("type"),
                "importance": c.get("importance", 0.5),
                "matched_evidence_count": len(claim_evidence)
            })

        # Construct instructions (static prefix)
        instructions = f"""You are the Spectrue Verdict Engine. Strictly evaluate claims against evidence.

Task:
1. Analyze EACH claim individually against the provided evidence.
2. For each claim, assign a `verdict_score` (0.0-1.0) and `verdict` (verified/refuted/unverified/partially_verified).
3. Provide a global `danger_score` (0.0-1.0) and `style_score` (0.0-1.0).
4. Explain the overall result in `rationale`.

Scores:
- 0.8-1.0: Verified (True)
- 0.6-0.8: Plausible (Mostly True/Likely)
- 0.4-0.6: Unverified (No clear evidence)
- 0.2-0.4: Misleading (False context)
- 0.0-0.2: Refuted (False)

Requirements:
- Aggregated `verified_score` will be calculated automatically from your claim scores, DO NOT output it manually.
- POLICY: Primary source requirement applies to each claim individually.
- CRITICAL: Write the `rationale` ENTIRELY in {lang_name} ({lang}).

Rationale Style Guide (User-Facing):
- Write for a general reader, NOT a system log. Use natural, simple language (1-2 paragraphs).
- FOCUS: Can this be trusted? Why or why not?
- FORBIDDEN: Do not use technical terms like "primary source", "independent domains", "score", "cap", "relevance", "deduplication", "JSON", "parameters".
- Instead of "lack of primary sources", say "no official confirmation found".
- Instead of "insufficient independent sources", say "verified by only one source".
- BAD: "Score limited to 0.6 due to lack of primary source."
- GOOD: "The claim is supported by Bloomberg, but there are no official statements confirming it yet. Therefore, it cannot be considered fully verified."

Output valid JSON:
{{
  "claim_verdicts": [
    {{
      "claim_id": "c1",
      "verdict_score": 0.9, 
      "verdict": "verified",
      "reason": "..." 
    }}
  ],
  "danger_score": 0.1,
  "style_score": 0.9,
  "explainability_score": 0.9,
  "rationale": "Overall summary..."
}}

You MUST respond in valid JSON.

{UNIVERSAL_METHODOLOGY_APPENDIX}
"""

        # Construct prompt (variable content)
        prompt = f"""Evaluate these claims based strictly on the Evidence.

Original Fact:
{pack.get("original_fact")}

Claims to Verify:
{claims_info}

Evidence (Grouped by Claim ID where possible):
{sources_by_claim}

Constraints (Caps):
Global Confidence Cap: {global_cap}
"""
        # Select reasoning effort
        effort = "medium" 
        
        # Cache key identifies static instructions for prefix caching
        cache_key = f"evidence_score_v4_{lang}"

        try:
            # Determine appropriate model
            m = model
            if "gpt-5" not in m and "o1" not in m:
                m = "gpt-5.2"

            # REQUIRED: The word "JSON" must appear in the INPUT message
            prompt += "\n\nReturn the result in JSON format."

            result = await self.llm_client.call_json(
                model=m,
                input=prompt,
                instructions=instructions,
                reasoning_effort=effort,
                cache_key=cache_key,
                timeout=float(self.runtime.llm.timeout_sec),
                trace_kind="score_evidence",
            )
            
            # --- Aggregation Logic (T3) ---
            claim_verdicts = result.get("claim_verdicts", [])
            
            # 1. Map scores to claims to get importance
            weighted_sum = 0.0
            total_importance = 0.0
            core_refuted = False
            
            # Lookup map for claims
            claims_map = {c["id"]: c for c in (pack.get("claims") or [])}
            
            for cv in claim_verdicts:
                cid = cv.get("claim_id")
                score = float(cv.get("verdict_score", 0.5))
                claim_obj = claims_map.get(cid)
                
                if claim_obj:
                    imp = float(claim_obj.get("importance", 0.5))
                    weighted_sum += score * imp
                    total_importance += imp
                    
                    # Veto: If a CORE claim is REFUTED (<0.2), global can't be true
                    if claim_obj.get("type") == "core" and score < 0.25:
                        core_refuted = True
                else:
                    # Fallback if ID mismatch
                    weighted_sum += score * 0.5
                    total_importance += 0.5

            # Calculate Global Verified Score
            if total_importance > 0:
                final_v_score = weighted_sum / total_importance
            else:
                final_v_score = 0.5
                
            # Apply Core Veto
            if core_refuted and final_v_score > 0.3:
                logger.info("[Scoring] Core claim refuted. Dragging score down to 0.25")
                final_v_score = 0.25
                result["rationale"] = f"[Core Claim Refuted] {result.get('rationale', '')}"

            result["verified_score"] = final_v_score
            
            # Post-clamp to ensure LLM respected the cap (T165)
            v_score = float(result.get("verified_score", 0.5))
            if v_score > global_cap:
                logger.info(f"[Scoring] Aggregated {v_score} > cap {global_cap}. Clamping.")
                result["verified_score"] = global_cap
                result["cap_enforced"] = True
            
            return result

        except Exception as e:
            logger.exception("[M48] Evidence scoring failed: %s", e)
            Trace.event("llm.error", {"kind": "score_evidence", "error": str(e)})
            return {
                "verified_score": 0.5,
                "confidence_score": 0.5,
                "rationale": "Error during evaluation."
            }

    def detect_evidence_gaps(self, pack: EvidencePack) -> list[str]:
        """
        Identify missing evidence types (T169).
        """
        # Can be logic-based or LLM-based.
        # Logic-based for speed/predictability (Nano usage optional).
        gaps = []
        metrics = pack.get("metrics", {})
        per_claim = metrics.get("per_claim", {})
        
        for cid, m in per_claim.items():
            if m.get("independent_domains", 0) < 2:
                gaps.append(f"Claim {cid}: Cited by fewer than 2 independent sources.")
            
            # Retrieve claim reqs
            # (Need to lookup claim object in pack)
            claim_obj = next((c for c in (pack.get("claims") or []) if c["id"] == cid), None)
            if claim_obj:
                req = claim_obj.get("evidence_requirement", {})
                if req.get("needs_primary_source") and not m.get("primary_present"):
                     gaps.append(f"Claim {cid}: Missing primary source.")
                if req.get("needs_quote_verification") and not m.get("quote_match"): # (Hypothetical field)
                     pass

    def _strip_internal_source_markers(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text or ""
        s = text
        s = re.sub(r"\[(?:TRUSTED|RAW)\]\s*", "", s)
        s = re.sub(r"\[REL=\s*\d+(?:\.\d+)?\]\s*", "", s)
        s = re.sub(r"\bTRUSTED\s*/\s*REL=\s*\d+(?:\.\d+)?\b", "", s)
        s = re.sub(r"\bREL=\s*\d+(?:\.\d+)?\b", "", s)
        s = re.sub(r"\s{2,}", " ", s).strip()
        return s

    def _maybe_drop_style_section(self, rationale: str, *, honesty_score: float | None, lang: str | None) -> str:
        if not rationale or not isinstance(rationale, str):
            return rationale or ""
        try:
            h = float(honesty_score) if honesty_score is not None else None
        except Exception:
            h = None
        if h is None or h < 0.80:
            return rationale

        lc = (lang or "").lower()
        if lc == "uk":
            labels = ["Стиль та контекст:"]
        elif lc == "ru":
            labels = ["Стиль и контекст:"]
        elif lc == "es":
            labels = ["Estilo y Contexto:"]
        elif lc == "de":
            labels = ["Stil und Kontext:"]
        elif lc == "fr":
            labels = ["Style et Contexte:"]
        elif lc == "ja":
            labels = ["文体と文脈:"]
        elif lc == "zh":
            labels = ["风格与语境:"]
        else:
            labels = ["Style and Context:"]

        s = rationale
        for label in labels:
            # Try to remove from label to end of string mostly
            s = re.sub(rf"\s*{re.escape(label)}.*$", "", s, flags=re.IGNORECASE | re.DOTALL)
            # Fallback cleanup if label was inline somehow (unlikely if at end)
            s = re.sub(rf"(?:^|\n)\s*{re.escape(label)}.*(?:\n|$)", "\n", s, flags=re.IGNORECASE)
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
        return s

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
                "verified_score": 0.5, "context_score": 0.5, "danger_score": 0.0,
                "style_score": 0.5, "confidence_score": 0.2, "rationale_key": "errors.agent_call_failed"
            }

        # Post-processing
        if "rationale" in result and isinstance(result["rationale"], dict):
             lines = []
             for k, v in result["rationale"].items():
                  lines.append(f"**{k.replace('_', ' ').title()}:** {v}")
             result["rationale"] = "\n\n".join(lines)
        elif "rationale" in result and isinstance(result["rationale"], list):
             result["rationale"] = "\n".join([str(item) for item in result["rationale"]])

        if "rationale" in result:
            result["rationale"] = self._strip_internal_source_markers(str(result["rationale"] or ""))
            try:
                cs = float(result.get("context_score", 0.5))
                ss = float(result.get("style_score", 0.5))
                honesty = (max(0.0, min(1.0, cs)) + max(0.0, min(1.0, ss))) / 2.0
            except Exception:
                honesty = None
            result["rationale"] = self._maybe_drop_style_section(result["rationale"], honesty_score=honesty, lang=lang)
            
        if "analysis" in result and isinstance(result["analysis"], str):
             result["analysis"] = self._strip_internal_source_markers(result["analysis"])

        return result
