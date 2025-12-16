from spectrue_core.verification.evidence_pack import EvidencePack
from .base_skill import BaseSkill, logger
from spectrue_core.utils.trace import Trace
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.utils.security import sanitize_input
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
        prompt = f"""Evaluate the Truthfulness of the Original Fact based strictly on the provided Evidence.

Original Fact:
{pack.get("original_fact")}

Claims & Evidence:
{sources_by_claim}

Constraints (Caps):
Global Confidence Cap: {global_cap} (You CANNOT exceed this score).
Reasons: {cap_reasons}

Task:
1. Assign `verified_score` (0.0-1.0): Truthfulness probability. MUST BE <= {global_cap}.
2. Assign `danger_score` (0.0-1.0): Harmfulness/toxicity.
3. Assign `style_score` (0.0-1.0): Level of manipulative language in original fact.
4. Assign `explainability_score` (0.0-1.0): How well the evidence explains the verdict (Alpha).
5. provide `rationale`: Clear explanation of the verdict, citing strong/weak evidence.
   - Mention if score was limited by evidence gaps (independent sources, primary sources).
   - If all sources repeat the same fact without independent confirmation, lower confidence.
   - Use language: {lang}.

Output valid JSON:
{{
  "verified_score": 0.8,
  "danger_score": 0.1,
  "style_score": 0.2,
  "explainability_score": 0.9,
  "rationale": "..."
}}
"""
        # Select reasoning effort
        effort = "medium" # T185 requirement
        
        # FIX: Cache key must depend on content, not just language!
        import hashlib
        import json
        
        # M54: Hash canonical content to avoid false hits
        hash_payload = {
            "fact": pack.get("original_fact"),
            "evidence": sources_by_claim,
            "constraints": {"global_cap": global_cap, "reasons": cap_reasons},
            "lang": lang,
        }
        canonical = json.dumps(hash_payload, sort_keys=True, default=str)
        content_hash = hashlib.md5(canonical.encode()).hexdigest()
        cache_key = f"evidence_score_v2_{lang}_{content_hash}"
        
        # M54: Debug trace for cache verification
        # M55: Debug trace for cache verification is now handled by LLMClient payload logging.

        try:
            # Determine appropriate model (use config or passed arg)
            m = model
            if "gpt-5" not in m and "o1" not in m:
                m = "gpt-5.2"

            result = await self.llm_client.call_json(
                model=m,
                input=prompt,
                instructions="You are the Spectrue Verdict Engine. Strictly evaluate claims against evidence.",
                reasoning_effort=effort,
                cache_key=cache_key,
                timeout=float(self.runtime.llm.timeout_sec),
                trace_kind="score_evidence",
            )
            
            # Post-clamp to ensure LLM respected the cap (T165)
            # This is also done in pipeline/verifier, but good to have here too.
            v_score = float(result.get("verified_score", 0.5))
            if v_score > global_cap:
                logger.info(f"[Scoring] LLM returned {v_score} > cap {global_cap}. Clamping.")
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
            s = re.sub(rf"(?:^|\n)\s*{re.escape(label)}.*(?:\n|$)", "\n", s, flags=re.IGNORECASE)
            s = re.sub(rf"\s*{re.escape(label)}.*$", "", s, flags=re.IGNORECASE)
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
            "ru": "ВАЖНО: Не упоминай в ответе служебные метки источников вроде [TRUSTED], [REL=…], [RAW].",
            "en": "IMPORTANT: Do not mention internal source tags like [TRUSTED], [REL=…], [RAW] in your output.",
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
