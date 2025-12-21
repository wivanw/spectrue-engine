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

# M70: Schema imports for structured scoring
from spectrue_core.schema import (
    ClaimUnit,
    StructuredVerdict,
    ClaimVerdict,
    AssertionVerdict,
    VerdictStatus,
    StructuredDebug,
)

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
Calculate the global `verified_score` using CORE DOMINANCE:

1. **CORE claims (importance >= 0.7)**: These drive the verdict. Weight them heavily.
2. **SIDE facts (importance < 0.7)**: Modifiers only, never exceed CORE.

**FORMULA**:
- If ALL core claims score >= 0.6: `verified_score = core_avg * 0.8 + side_avg * 0.2`
- If ANY core claim scores < 0.4: `verified_score = min(core_avg, 0.5)` (Cap: weak core = weak total)
- If NO core claims exist: Use simple weighted average

**CRITICAL**: If the main thesis is unverified, side facts CANNOT save the global score.

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
            safe_content = sanitize_input(raw_content)[:MAX_SNIPPET_LEN]
            
            key_snippet = e.get("key_snippet") or e.get("quote")
            stance = e.get("stance", "MENTION")
            
            if key_snippet and stance in ["SUPPORT", "REFUTE", "MIXED"]:
                safe_quote = sanitize_input(key_snippet)[:MAX_QUOTE_LEN]
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
        
        # --- 3. SYSTEM PROMPT (Schema-First) ---
        instructions = f"""You are the Spectrue Schema-First Verdict Engine.
Your task is to score each ASSERTION individually, then aggregate to claim and global verdicts.

# CRITICAL RULES

## DIMENSION HANDLING
1. **FACT assertions**: Strictly verify. Can be VERIFIED, REFUTED, or AMBIGUOUS.
2. **CONTEXT assertions**: Soft verify. Only VERIFIED or AMBIGUOUS. Rarely REFUTED.
3. **🚨 GOLDEN RULE**: CONTEXT evidence CANNOT refute FACT assertions!
   - If time_reference says "Ukraine time" but event is in "Miami" → location is still a FACT
   - Time context doesn't contradict location facts

## SCORING SCALE (0.0 - 1.0)
- **0.8 - 1.0**: Verified (strong evidence confirms)
- **0.6 - 0.8**: Likely verified (supported but not definitive)
- **0.4 - 0.6**: Ambiguous (insufficient evidence)
- **0.2 - 0.4**: Unlikely (evidence suggests doubt)
- **0.0 - 0.2**: Refuted (evidence contradicts)

## AGGREGATION
1. Score each assertion independently
2. Claim verdict = importance-weighted mean of FACT assertion scores
3. CONTEXT assertions are modifiers, not drivers
4. Global verified_score = importance-weighted mean of claim verdicts

## CONTENT_UNAVAILABLE Handling
If evidence has `content_status: "unavailable"`:
- Lower explainability_score (we couldn't read the source)
- Assertion stays AMBIGUOUS, NOT refuted
- This is NOT evidence against the claim

# OUTPUT FORMAT
```json
{{
  "claim_verdicts": [
    {{
      "claim_id": "c1",
      "verdict_score": 0.85,
      "status": "verified",
      "assertion_verdicts": [
        {{
          "assertion_key": "event.location.city",
          "dimension": "FACT",
          "score": 0.9,
          "status": "verified",
          "evidence_count": 2,
          "rationale": "Multiple sources confirm Miami"
        }},
        {{
          "assertion_key": "event.time_reference",
          "dimension": "CONTEXT",
          "score": 0.8,
          "status": "verified",
          "evidence_count": 1,
          "rationale": "Time zone context verified"
        }}
      ],
      "reason": "Location and timing confirmed by official sources."
    }}
  ],
  "verified_score": 0.85,
  "explainability_score": 0.9,
  "danger_score": 0.1,
  "style_score": 0.9,
  "rationale": "Global summary in {lang_name}..."
}}
```

Write rationale and reason in **{lang_name}** ({lang}).
Return valid JSON.
"""

        # --- 4. USER PROMPT ---
        prompt = f"""Score these claims with per-assertion verdicts.

Claims with Assertions:
{json.dumps(claims_data, indent=2, ensure_ascii=False)}

Evidence by Assertion:
{json.dumps(evidence_by_assertion, indent=2, ensure_ascii=False)}

Remember: CONTEXT cannot refute FACT. Score each assertion independently.
Return JSON.
"""

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
        def safe_score(val, default=-1.0) -> float:
            try:
                f = float(val)
            except (TypeError, ValueError):
                return default
            if default < 0 and (f < 0.0 or f > 1.0):
                return default
            return max(0.0, min(1.0, f))

        def parse_status(s: str) -> VerdictStatus:
            s_lower = (s or "").lower()
            if s_lower in ("verified", "confirmed"):
                return VerdictStatus.VERIFIED
            elif s_lower in ("refuted", "false"):
                return VerdictStatus.REFUTED
            elif s_lower == "partially_verified":
                return VerdictStatus.PARTIALLY_VERIFIED
            elif s_lower == "unverified":
                return VerdictStatus.UNVERIFIED
            else:
                return VerdictStatus.AMBIGUOUS

        # Parse claim verdicts
        claim_verdicts: list[ClaimVerdict] = []
        raw_claims = raw.get("claim_verdicts", [])
        
        if isinstance(raw_claims, list):
            for rc in raw_claims:
                if not isinstance(rc, dict):
                    continue

                # Parse assertion verdicts
                assertion_verdicts: list[AssertionVerdict] = []
                raw_assertions = rc.get("assertion_verdicts", [])
                
                fact_verified = 0
                fact_total = 0
                
                if isinstance(raw_assertions, list):
                    for ra in raw_assertions:
                        if not isinstance(ra, dict):
                            continue
                        
                        dim = (ra.get("dimension") or "FACT").upper()
                        if dim == "FACT":
                            fact_total += 1
                            score = safe_score(ra.get("score"), default=-1.0)
                            if score >= 0.6:
                                fact_verified += 1
                        
                        assertion_verdicts.append(AssertionVerdict(
                            assertion_key=ra.get("assertion_key", ""),
                            dimension=dim,
                            status=parse_status(ra.get("status", "")),
                            score=safe_score(ra.get("score"), default=-1.0),
                            evidence_count=int(ra.get("evidence_count", 0)),
                            supporting_urls=ra.get("supporting_urls", []),
                            rationale=ra.get("rationale", ""),
                        ))

                claim_verdicts.append(ClaimVerdict(
                    claim_id=rc.get("claim_id", ""),
                    status=parse_status(rc.get("status", "")),
                    verdict_score=safe_score(rc.get("verdict_score"), default=-1.0),
                    assertion_verdicts=assertion_verdicts,
                    evidence_count=int(rc.get("evidence_count", 0)),
                    fact_assertions_verified=fact_verified,
                    fact_assertions_total=fact_total,
                    reason=rc.get("reason", ""),
                    key_evidence=rc.get("key_evidence", []),
                ))

        # Parse global scores
        verified = safe_score(raw.get("verified_score"), default=-1.0)
        explainability = safe_score(raw.get("explainability_score"), default=-1.0)
        danger = safe_score(raw.get("danger_score"), default=-1.0)
        style = safe_score(raw.get("style_score"), default=-1.0)

        # Fallback: if verified_score missing, calculate from claim verdicts
        if verified < 0 and claim_verdicts:
            scores = [(cv.verdict_score, 1.0) for cv in claim_verdicts if cv.verdict_score >= 0]
            if scores:
                total_weight = sum(w for _, w in scores)
                if total_weight > 0:
                    verified = sum(s * w for s, w in scores) / total_weight

        # Clean rationale
        rationale = self._strip_internal_source_markers(str(raw.get("rationale", "")))
        rationale = self._maybe_drop_style_section(rationale, honesty_score=style, lang=lang)

        # Build debug info
        debug = StructuredDebug(
            per_claim={cv.claim_id: {"score": cv.verdict_score, "status": cv.status.value} for cv in claim_verdicts},
            content_unavailable_count=0,  # TODO: count from evidence
        )

        return StructuredVerdict(
            claim_verdicts=claim_verdicts,
            verified_score=verified,
            explainability_score=explainability,
            danger_score=danger,
            style_score=style,
            rationale=rationale,
            structured_debug=debug,
            overall_confidence=verified,  # Legacy compat
        )
