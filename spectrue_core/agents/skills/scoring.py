from spectrue_core.verification.evidence_pack import EvidencePack
from .base_skill import BaseSkill, logger
from spectrue_core.utils.trace import Trace
from spectrue_core.constants import SUPPORTED_LANGUAGES
from spectrue_core.utils.security import sanitize_input
from spectrue_core.agents.prompts import get_prompt
from datetime import datetime
import json
import hashlib
import re

# Constants for input safety (Defensive Programming)
MAX_SNIPPET_LEN = 600
MAX_QUOTE_LEN = 300

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
            safe_snippet = sanitize_input(raw_snippet)[:MAX_SNIPPET_LEN]
            
            key_snippet = r.get("key_snippet")
            stance = r.get("stance", "")
            
            # Visual cue: Highlight quotes found by Clustering
            if key_snippet and stance in ["SUPPORT", "REFUTE", "MIXED"]:
                safe_key = sanitize_input(key_snippet)[:MAX_QUOTE_LEN]
                content_text = f'📌 QUOTE: "{safe_key}"\nℹ️ CONTEXT: {safe_snippet}'
            else:
                content_text = safe_snippet
            
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

        # --- 3. SYSTEM PROMPT (The Constitution) ---
        instructions = f"""You are the Spectrue Verdict Engine.
Your task is to classify the reliability of claims based *strictly* on the provided Evidence.

# INPUT DATA
- **Claims**: Note the `importance` (0.0-1.0). High importance = Core Thesis.
- **Evidence**: Look for "📌 QUOTE" segments. `stance` is a hint; always verify against the quote/excerpt.
- **Metadata**: `source_reliability_hint` is context, not a hard rule.

# SCORING SCALE (0.0 - 1.0)
- **0.8 - 1.0 (Verified)**: Strong confirmation (direct quotes, official consensus).
- **0.6 - 0.8 (Plausible)**: Supported, but may lack direct/official confirmation or deep detail.
- **0.4 - 0.6 (Ambiguous)**: Insufficient, irrelevant, or conflicting evidence. DO NOT GUESS. Absence of evidence is not False.
- **0.2 - 0.4 (Unlikely)**: Evidence suggests the claim is doubtful.
- **0.0 - 0.2 (Refuted)**: Evidence contradicts the claim.

# AGGREGATION LOGIC (Global Score)
Calculate the global `verified_score` yourself:
- **Core Claims (High Importance)** drive the verdict.
- **Side Facts** are modifiers.
- If the core claim is unverified, the global score must reflect that uncertainty.

# WRITING GUIDELINES (User-Facing Text Only)
- Write `rationale` and `reason` ENTIRELY in **{lang_name}** ({lang}).
- **Tone**: Natural, journalistic style.
- **FORBIDDEN TERMS (in text/rationale)**: Do not use technical words like "JSON", "dataset", "primary source", "relevance score", "cap" in the readable text. 
  - *Allowed in JSON keys/values, but forbidden in human explanations.*
  - Instead of "lack of primary source", say "no official confirmation found".

# OUTPUT FORMAT
Return valid JSON:
{{
  "claim_verdicts": [
    {{"claim_id": "c1", "verdict_score": 0.9, "verdict": "verified", "reason": "..."}}
  ],
  "verified_score": 0.85,
  "explainability_score": 0.8,
  "danger_score": 0.1,
  "style_score": 0.9,
  "rationale": "Global summary in {lang_name}..."
}}

# GLOBAL SCORES EXPLANATION
- **verified_score**: The GLOBAL aggregated truthfulness, calculated by YOU based on importance.
- **explainability_score** (0.0-1.0): How well the evidence supports your rationale. 1.0 = every claim verdict is backed by direct quotes.
- **danger_score** (0.0-1.0): How harmful if acted upon? (0.0 = harmless, 1.0 = dangerous misinformation)
- **style_score** (0.0-1.0): How neutral is the writing style? (0.0 = heavily biased, 1.0 = neutral journalism)
"""

        # --- 4. USER PROMPT (The Data) ---
        prompt = f"""Evaluate these claims based strictly on the Evidence.

Original Fact:
{safe_original_fact}

Claims to Verify:
{json.dumps(claims_info, indent=2, ensure_ascii=False)}

Evidence:
{json.dumps(sources_by_claim, indent=2, ensure_ascii=False)}

Return JSON.
"""
        
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
            
            # --- 5. TYPE SAFETY & CLAMPING (Defensive) ---
            
            def safe_score(val, default=-1.0):
                try:
                    f = float(val)
                except (TypeError, ValueError):
                    return default
                
                # Preserve sentinel semantics: if default < 0 and value outside [0..1], return sentinel
                if default < 0 and (f < 0.0 or f > 1.0):
                    return default
                
                return max(0.0, min(1.0, f))

            # Clamp Global Scores (-1.0 = "LLM didn't provide this")
            if "verified_score" in result:
                result["verified_score"] = safe_score(result["verified_score"], default=-1.0)
            else:
                # Fallback aggregation only if LLM forgot the field
                verdicts = result.get("claim_verdicts")
                if isinstance(verdicts, list) and verdicts:
                    vals = []
                    for v in verdicts:
                        if isinstance(v, dict):
                            vals.append(safe_score(v.get("verdict_score", 0.5)))
                    
                    if vals:
                        result["verified_score"] = sum(vals) / len(vals)
                    else:
                        result["verified_score"] = -1.0
                else:
                    result["verified_score"] = -1.0
            
            result["explainability_score"] = safe_score(result.get("explainability_score"), default=-1.0)
            result["danger_score"] = safe_score(result.get("danger_score"), default=-1.0)
            result["style_score"] = safe_score(result.get("style_score"), default=-1.0)

            # Clamp Individual Verdicts
            claim_verdicts = result.get("claim_verdicts")
            if isinstance(claim_verdicts, list):
                for cv in claim_verdicts:
                    if isinstance(cv, dict):
                        cv["verdict_score"] = safe_score(cv.get("verdict_score", 0.5))
            else:
                result["claim_verdicts"] = []

            return result

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
            s = re.sub(rf"\s*{re.escape(label)}.*$", "", s, flags=re.IGNORECASE | re.DOTALL)
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
