# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

import os
import json
import logging
import re
from openai import AsyncOpenAI
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.utils.security import sanitize_input
from spectrue_core.config import SpectrueConfig
import asyncio
from datetime import datetime
from spectrue_core.utils.runtime import is_local_run
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)

def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y", "on")

ENGINE_DEBUG = _env_flag("SPECTRUE_ENGINE_DEBUG")
LOG_PROMPTS = _env_flag("SPECTRUE_ENGINE_LOG_PROMPTS")


class FactCheckerAgent:
    def __init__(self, config: SpectrueConfig = None):
        self.config = config
        api_key = config.openai_api_key if config else os.getenv("OPENAI_API_KEY")
        
        self.client = AsyncOpenAI(api_key=api_key)
        
        try:
            t = float(os.getenv("OPENAI_TIMEOUT", "60"))
            if not (5 <= t <= 300):
                t = 60.0
        except Exception:
            t = 60.0
        self.timeout = t
        try:
            c = int(os.getenv("OPENAI_CONCURRENCY", "6"))
        except Exception:
            c = 6
        c = max(1, min(c, 16))
        self._sem = asyncio.Semaphore(c)
        self.last_query_meta: dict = {}

    def _canonicalize_action(self, action: str | None) -> str:
        """
        Normalize action verbs to a small canonical set.

        This is used as an explicit anchor for query generation and for downstream scoring heuristics.
        """
        a = (action or "").strip().lower()
        if not a:
            return "claim"

        # Normalize common "strong" verbs first (ban/prohibit/restrict family).
        strong_map = {
            # EN
            "ban": "ban",
            "banned": "ban",
            "prohibit": "prohibit",
            "prohibited": "prohibit",
            "forbid": "forbid",
            "forbidden": "forbid",
            "restrict": "restrict",
            "restricted": "restrict",
            # UK
            "заборона": "ban",
            "заборонили": "ban",
            "заборонено": "ban",
            "заборонити": "ban",
            # RU
            "запрет": "ban",
            "запретили": "ban",
            "запрещено": "ban",
            "запретить": "ban",
        }
        for k, v in strong_map.items():
            if k in a:
                return v

        # A small set of other common actions.
        if any(k in a for k in ("remove", "removed", "take down", "told to remove", "asked to remove")):
            return "remove"
        if any(k in a for k in ("arrest", "arrested", "detain", "detained")):
            return "arrest"
        if any(k in a for k in ("close", "closed", "shut down", "shutdown")):
            return "close"

        # Default: keep a compact cleaned verb phrase (bounded).
        a = re.sub(r"\s+", " ", a).strip()
        return a[:32] if len(a) > 32 else a

    def _normalize_claim_decomposition(self, obj: dict | None, *, fact: str) -> dict:
        """
        Ensure the claim decomposition structure is present and normalized.

        Required shape (fields must exist; strings or null):
        {
          "subject": string,
          "action": string,
          "object": string,
          "where": string | null,
          "when": string | null,
          "by_whom": string | null
        }
        """
        raw = obj if isinstance(obj, dict) else {}

        def _clean_str(v: object, *, max_len: int) -> str:
            if v is None:
                return ""
            if not isinstance(v, str):
                v = str(v)
            s = re.sub(r"\s+", " ", v).strip()
            return s[:max_len] if len(s) > max_len else s

        def _clean_opt(v: object, *, max_len: int) -> str | None:
            if v is None:
                return None
            s = _clean_str(v, max_len=max_len)
            return s if s else None

        subject = _clean_str(raw.get("subject"), max_len=160)
        action = self._canonicalize_action(_clean_str(raw.get("action"), max_len=64))
        obj_text = _clean_str(raw.get("object"), max_len=220)

        # If the LLM returned empty strings, keep decomposition stable by falling back to the fact itself.
        if not subject and isinstance(fact, str) and fact.strip():
            subject = _clean_str(fact.strip(), max_len=80)
        if not obj_text and isinstance(fact, str) and fact.strip():
            obj_text = _clean_str(fact.strip(), max_len=120)

        return {
            "subject": subject,
            "action": action or "claim",
            "object": obj_text,
            "where": _clean_opt(raw.get("where"), max_len=120),
            "when": _clean_opt(raw.get("when"), max_len=120),
            "by_whom": _clean_opt(raw.get("by_whom"), max_len=140),
        }

    def _action_anchor_for_lang(self, canonical_action: str, lang_code: str) -> str:
        """
        Provide a compact action anchor term for the given language.
        """
        lc = (lang_code or "en").lower()
        a = self._canonicalize_action(canonical_action)
        if a not in ("ban", "prohibit", "forbid", "restrict"):
            return a

        # Minimal in-code translations for anchoring only.
        map_by_lang = {
            "en": {"ban": "ban", "prohibit": "prohibit", "forbid": "forbid", "restrict": "restrict"},
            "uk": {"ban": "заборонили", "prohibit": "заборонили", "forbid": "заборонили", "restrict": "обмежили"},
            "ru": {"ban": "запретили", "prohibit": "запретили", "forbid": "запретили", "restrict": "ограничили"},
            "de": {"ban": "verbot", "prohibit": "verbot", "forbid": "verbot", "restrict": "beschränkt"},
            "es": {"ban": "prohibió", "prohibit": "prohibió", "forbid": "prohibió", "restrict": "restringió"},
            "fr": {"ban": "interdit", "prohibit": "interdit", "forbid": "interdit", "restrict": "restreint"},
            "ja": {"ban": "禁止", "prohibit": "禁止", "forbid": "禁止", "restrict": "制限"},
            "zh": {"ban": "禁止", "prohibit": "禁止", "forbid": "禁止", "restrict": "限制"},
        }
        return map_by_lang.get(lc, map_by_lang["en"]).get(a, a)

    def _probe_phrases(self, *, lang_code: str, missing: list[str]) -> list[str]:
        """
        Deterministic context probes used when claim attributes are missing.
        These are intentionally short to avoid bloating queries.
        """
        lc = (lang_code or "en").lower()
        probes_by_lang = {
            "en": {
                "where": ["venue security", "concert venue"],
                "when": ["when did it happen", "date"],
                "by_whom": ["organizers said", "venue said"],
            },
            "uk": {
                "where": ["охорона майданчика", "концертний майданчик"],
                "when": ["коли це сталося", "дата"],
                "by_whom": ["організатори сказали", "заява майданчика"],
            },
            "ru": {
                "where": ["охрана площадки", "концертная площадка"],
                "when": ["когда это было", "дата"],
                "by_whom": ["организаторы сказали", "заявление площадки"],
            },
            "de": {
                "where": ["Veranstaltungsort", "Sicherheitsdienst"],
                "when": ["wann passiert", "Datum"],
                "by_whom": ["Veranstalter sagten", "Aussage des Ortes"],
            },
            "es": {
                "where": ["lugar del concierto", "seguridad del recinto"],
                "when": ["cuándo ocurrió", "fecha"],
                "by_whom": ["organizadores dijeron", "el recinto dijo"],
            },
            "fr": {
                "where": ["lieu du concert", "sécurité du lieu"],
                "when": ["quand cela s'est produit", "date"],
                "by_whom": ["les organisateurs ont dit", "le lieu a dit"],
            },
            "ja": {
                "where": ["会場警備", "コンサート会場"],
                "when": ["いつ起きた", "日付"],
                "by_whom": ["主催者の声明", "会場の声明"],
            },
            "zh": {
                "where": ["场馆安保", "演唱会场馆"],
                "when": ["发生时间", "日期"],
                "by_whom": ["主办方称", "场馆方称"],
            },
        }
        table = probes_by_lang.get(lc, probes_by_lang["en"])
        out: list[str] = []
        for k in ("where", "when", "by_whom"):
            if k in missing:
                out.extend(table.get(k, []))
        # Bound and dedupe
        seen = set()
        uniq: list[str] = []
        for p in out:
            p = (p or "").strip()
            if not p:
                continue
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq[:4]

    def build_strict_queries(self, claim_decomposition: dict | None, *, lang: str, content_lang: str | None) -> list[str]:
        """
        Deterministic "strict anchoring" queries used for a refined search pass.

        This intentionally avoids an additional LLM call; it is a real strategy change when relevance is low.
        """
        claim = self._normalize_claim_decomposition(claim_decomposition, fact="")
        content_lang = content_lang or lang

        def _build_for(code: str) -> str:
            action_anchor = self._action_anchor_for_lang(claim.get("action"), code)
            missing = [k for k in ("where", "when", "by_whom") if claim.get(k) in (None, "")]
            probes = self._probe_phrases(lang_code=code, missing=missing)

            parts = [claim.get("subject") or "", action_anchor or "", claim.get("object") or ""]
            # Include available specifics to tighten the search.
            for k in ("where", "when", "by_whom"):
                v = claim.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            # If key specifics are missing, force probes to avoid context-poor "headline-only" queries.
            parts.extend(probes[:2])

            q = " ".join([p for p in parts if p]).strip()
            q = re.sub(r"\s+", " ", q).strip()
            return q[:150] if len(q) > 150 else q

        queries: list[str] = []
        queries.append(_build_for("en"))
        if content_lang != "en":
            queries.append(_build_for(content_lang))
        if lang not in ("en", content_lang):
            queries.append(_build_for(lang))
        return queries

    def _strip_internal_source_markers(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return text or ""
        s = text
        # Remove inline source markers that are meant for the model only.
        s = re.sub(r"\[(?:TRUSTED|RAW)\]\s*", "", s)
        s = re.sub(r"\[REL=\s*\d+(?:\.\d+)?\]\s*", "", s)
        # Remove common phrasing that leaks internal tags into the user-facing rationale.
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
        # Keep style/manipulation notes only when honesty is meaningfully impacted.
        if h is None or h < 0.80:
            return rationale

        labels: list[str]
        lc = (lang or "").lower()
        if lc == "uk":
            labels = ["Маніпуляції/Стиль:"]
        elif lc == "ru":
            labels = ["Манипуляции/Стиль:"]
        elif lc == "es":
            labels = ["Manipulación/Estilo:"]
        elif lc == "de":
            labels = ["Manipulation/Stil:", "Manipulations-/Stil:"]
        elif lc == "fr":
            labels = ["Manipulation/Style:"]
        elif lc == "ja":
            labels = ["操作/文体:"]
        elif lc == "zh":
            labels = ["操纵/文风:"]
        else:
            labels = ["Manipulation/Style:"]

        s = rationale
        for label in labels:
            # Remove a full labeled line if rationale is multi-line.
            s = re.sub(rf"(?:^|\n)\s*{re.escape(label)}.*(?:\n|$)", "\n", s, flags=re.IGNORECASE)
            # Remove a trailing inline segment if rationale is a single paragraph.
            s = re.sub(rf"\s*{re.escape(label)}.*$", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
        return s

    async def analyze(self, fact: str, context: str, gpt_model: str, lang: str, analysis_mode: str = "general") -> dict:
        """Анализирует факт на основе контекста и возвращает структурированный JSON."""

        prompt_key = 'prompts.aletheia_lite' if analysis_mode == 'lite' else 'prompts.final_analysis'
        prompt_template = get_prompt(lang, prompt_key)
        
        # Fallback to default prompt if lite is not available or empty
        if not prompt_template and analysis_mode == 'lite':
             prompt_template = get_prompt(lang, 'prompts.final_analysis')

        # Wrap inputs in tags and sanitize to prevent injection
        safe_fact = f"<statement>{sanitize_input(fact)}</statement>"
        safe_context = f"<context>{sanitize_input(context)}</context>"
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        # date kwarg is ignored if not present in template, safe to add
        prompt = prompt_template.format(fact=safe_fact, context=safe_context, date=current_date)

        # Prevent leaking internal source markers (added for model routing/scoring) into user-facing output.
        # This is in addition to post-processing sanitization as a defense-in-depth measure.
        no_tag_note_by_lang = {
            "uk": "ВАЖЛИВО: Не згадуй у відповіді службові мітки джерел на кшталт [TRUSTED], [REL=…], [RAW].",
            "ru": "ВАЖНО: Не упоминай в ответе служебные метки источников вроде [TRUSTED], [REL=…], [RAW].",
            "en": "IMPORTANT: Do not mention internal source tags like [TRUSTED], [REL=…], [RAW] in your output.",
        }
        prompt += "\n\n" + no_tag_note_by_lang.get((lang or "").lower(), no_tag_note_by_lang["en"])
        Trace.event(
            "llm.prompt",
            {
                "kind": "analysis",
                "model": gpt_model,
                "lang": lang,
                "analysis_mode": analysis_mode,
                "prompt_chars": len(prompt),
                "prompt": prompt,
            },
        )

        if LOG_PROMPTS:
            logger.debug("--- Prompt for %s (%s) ---\n%s\n--- End Prompt ---", gpt_model, lang, prompt)
        elif ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Built prompt for %s (%s): %d chars", gpt_model, lang, len(prompt))
        
        messages = [{"role": "user", "content": prompt}]

        def _max_output_tokens_for_mode(mode: str) -> int:
            key = "SPECTRUE_LLM_MAX_OUTPUT_TOKENS_GENERAL"
            default = 900
            if mode == "lite":
                key = "SPECTRUE_LLM_MAX_OUTPUT_TOKENS_LITE"
                default = 500
            elif mode == "deep":
                key = "SPECTRUE_LLM_MAX_OUTPUT_TOKENS_DEEP"
                default = 1100
            try:
                v = int(os.getenv(key, str(default)))
            except Exception:
                v = default
            return max(200, min(v, 4000))

        max_out = _max_output_tokens_for_mode(analysis_mode)

        # Temperature configuration for minimal creativity (strict factual analysis)
        # gpt-5-nano doesn't support temperature parameter (as of 2025-11)
        # gpt-5-nano/mini and o1 models don't support temperature parameter (or require default 1)
        models_without_temperature = ["gpt-5-nano", "gpt-5-mini", "gpt-5", "o1-mini", "o1-preview"]
        use_temperature = gpt_model not in models_without_temperature and not gpt_model.startswith("o1-")
        
        api_params = {
            "model": gpt_model,
            "response_format": {"type": "json_object"},
            "messages": messages,
            "timeout": self.timeout
        }

        # Output token budgeting (cost/latency control).
        # Reasoning-capable models use max_completion_tokens; other chat models use max_tokens.
        if "gpt-5" in gpt_model or gpt_model.startswith("o1-") or "o1" in gpt_model:
            api_params["max_completion_tokens"] = max_out
        else:
            api_params["max_tokens"] = max_out
        
        # Optimize speed for reasoning models
        if "gpt-5" in gpt_model or "o1" in gpt_model:
            api_params["reasoning_effort"] = "low"
            # GPT-5 and o1 models don't support temperature parameter
            use_temperature = False
        
        # Add temperature only if supported by the model
        if use_temperature:
            api_params["temperature"] = 0  # Deterministic for fact-checking

        try:
            if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[GPT] Calling %s with %d chars prompt...", gpt_model, len(prompt))
            async with self._sem:
                response = await self.client.chat.completions.create(**api_params)
            
            raw_content = response.choices[0].message.content
            Trace.event(
                "llm.response",
                {
                    "kind": "analysis",
                    "model": gpt_model,
                    "content_chars": len(raw_content or ""),
                    "content": raw_content,
                },
            )
            if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[GPT] Got response: %d chars", len(raw_content))
                logger.debug("[GPT] Response preview: %s...", raw_content[:300])
            
            # Parse the JSON response
            result = json.loads(raw_content)
            Trace.event(
                "llm.parsed",
                {
                    "kind": "analysis",
                    "model": gpt_model,
                    "keys": list(result.keys()),
                },
            )
            if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[GPT] ✓ JSON parsed successfully. Keys: %s", list(result.keys()))
            
            # Handle Chain-of-Thought filtering (support both keys)
            thought_process = result.pop("thought_process", None) or result.pop("_thought_process", None)
            if thought_process and ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[Aletheia-X Thought]: %s", thought_process)
            
            # Fix rationale if it's an object (Radical Candor prompt side-effect)
            if "rationale" in result and isinstance(result["rationale"], (dict, list)):
                if isinstance(result["rationale"], dict):
                    # Convert dict to formatted string
                    lines = []
                    for k, v in result["rationale"].items():
                        # Capitalize key for better display
                        key_display = k.replace('_', ' ').title()
                        lines.append(f"**{key_display}:** {v}")
                    result["rationale"] = "\n\n".join(lines)
                elif isinstance(result["rationale"], list):
                     result["rationale"] = "\n".join([str(item) for item in result["rationale"]])

            # Never leak internal source tags into user-facing strings.
            if "rationale" in result:
                result["rationale"] = self._strip_internal_source_markers(str(result["rationale"] or ""))
                try:
                    context_score = float(result.get("context_score", 0.5))
                    style_score = float(result.get("style_score", 0.5))
                    honesty = (max(0.0, min(1.0, context_score)) + max(0.0, min(1.0, style_score))) / 2.0
                except Exception:
                    honesty = None
                result["rationale"] = self._maybe_drop_style_section(
                    result["rationale"], honesty_score=honesty, lang=lang
                )
            if "analysis" in result and isinstance(result["analysis"], str):
                result["analysis"] = self._strip_internal_source_markers(result["analysis"])
            
            if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[GPT] Scores: V=%.2f, D=%.2f, Conf=%.2f",
                    float(result.get("verified_score", 0) or 0),
                    float(result.get("danger_score", 0) or 0),
                    float(result.get("confidence_score", 0) or 0),
                )
            return result
        except json.JSONDecodeError as e:
            logger.warning("[GPT] ✗ JSON parse error: %s", e)
            Trace.event("llm.parse_error", {"kind": "analysis", "model": gpt_model, "error": str(e)})
            if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[GPT] Raw content was: %s",
                    raw_content[:500] if 'raw_content' in locals() else 'N/A',
                )
            return {
                "verified_score": 0.5,
                "context_score": 0.5,
                "danger_score": 0.0,
                "style_score": 0.5,
                "confidence_score": 0.2,
                "rationale_key": "errors.agent_call_failed"
            }
        except Exception as e:
            logger.exception("[GPT] ✗ Error calling %s: %s", gpt_model, e)
            Trace.event("llm.error", {"kind": "analysis", "model": gpt_model, "error": str(e)})
            # Возвращаем дефолтные значения, включая низкую уверенность
            return {
                "verified_score": 0.5,
                "context_score": 0.5,
                "danger_score": 0.0,
                "style_score": 0.5,
                "confidence_score": 0.2, # <-- Низкая уверенность при ошибке
                "rationale_key": "errors.agent_call_failed"
            }

    async def generate_search_queries(
        self, 
        fact: str, 
        context: str = "", 
        lang: str = "en",
        content_lang: str = None  # M31: Content language from detection
    ) -> list[str]:
        """
        Generate search queries using GPT-5 Nano.
        
        M31: 3-tier query generation:
        - Query 1: Always English (global coverage)
        - Query 2: Content language if != EN (source language)
        - Query 3: UI language if != EN and != content_lang (user preference)
        
        Returns:
            List of 2-3 queries depending on language overlap
        """
        import json
        import re
        
        # M31: Determine which languages to generate
        content_lang = content_lang or lang  # Fallback to UI lang
        
        languages_needed = []
        
        # Always include English
        languages_needed.append(("en", "English", "global"))
        
        # Add content language if different from English
        if content_lang != "en":
            lang_names = {
                "en": "English", "uk": "Ukrainian", "ru": "Russian",
                "de": "German", "es": "Spanish", "fr": "French",
                "ja": "Japanese", "zh": "Chinese"
            }
            lang_name = lang_names.get(content_lang, content_lang.title())
            languages_needed.append((content_lang, lang_name, "content"))
        
        # Add UI language if different from both
        if lang not in ["en", content_lang]:
            lang_names = {
                "en": "English", "uk": "Ukrainian", "ru": "Russian",
                "de": "German", "es": "Spanish", "fr": "French",
                "ja": "Japanese", "zh": "Chinese"
            }
            lang_name = lang_names.get(lang, lang.title())
            languages_needed.append((lang, lang_name, "ui"))
        
        # Limit to 3 queries max
        languages_needed = languages_needed[:3]
        if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[M31] Generating %d queries for languages: %s", len(languages_needed), [l[0] for l in languages_needed])
        
        # Clean input
        clean_fact = re.sub(r'http\S+', '', fact[:800]).replace("\n", " ").strip()
        clean_context = context[:500].replace("\n", " ").strip() if context else ""
        
        # Build prompt for multi-language generation
        language_instructions = "\n".join([
            f"{i+1}. {name} ({code}) - {purpose} search"
            for i, (code, name, purpose) in enumerate(languages_needed)
        ])
        
        async def _call_query_llm(prompt_text: str, *, kind: str) -> dict:
            Trace.event(
                "llm.prompt",
                {
                    "kind": kind,
                    "model": "gpt-5-nano",
                    "lang": lang,
                    "content_lang": content_lang,
                    "prompt_chars": len(prompt_text),
                    "prompt": prompt_text,
                },
            )
            async with self._sem:
                response = await self.client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[{"role": "user", "content": prompt_text}],
                    response_format={"type": "json_object"},
                    timeout=30,
                )
            raw = response.choices[0].message.content
            Trace.event(
                "llm.response",
                {
                    "kind": kind,
                    "model": "gpt-5-nano",
                    "content_chars": len(raw or ""),
                    "content": raw,
                },
            )
            result = json.loads(raw)
            Trace.event(
                "llm.parsed",
                {
                    "kind": kind,
                    "model": "gpt-5-nano",
                    "keys": list(result.keys()),
                },
            )
            return result

        # Single LLM call: queries + claim decomposition.
        prompt_queries = f"""Decompose the claim into a structured object and generate exactly {len(languages_needed)} search queries to fact-check it.

Rules:
1) Output MUST be valid JSON (no markdown).
2) Claim decomposition MUST include these keys exactly:
   subject, action, object, where, when, by_whom
   - subject/action/object are strings (can be empty, but should be best-effort).
   - where/when/by_whom are either strings or null.
3) action MUST be normalized to a verb anchor (prefer: ban / prohibit / forbid / restrict; otherwise a short verb).
4) Each query MUST include explicit anchors derived from decomposition:
   - subject
   - action (as a verb anchor)
   - object
5) If where/when/by_whom are null, queries MUST include contextual probes (e.g., "venue security", "organizers said", "when did it happen").
6) 5-12 words per query, no URLs.

LANGUAGES NEEDED:
{language_instructions}

Output JSON only:
{{
  "claim": {{
    "subject": "…",
    "action": "ban",
    "object": "…",
    "where": null,
    "when": null,
    "by_whom": null
  }},
  "queries": [{", ".join([f'"{lang[1].lower()} query"' for lang in languages_needed])}]
}}

CLAIM: {clean_fact}
CONTEXT: {clean_context if clean_context else "None"}"""

        try:
            if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[GPT-5-Nano] Generating %d queries for: %s...", len(languages_needed), clean_fact[:80])

            result_queries = await _call_query_llm(prompt_queries, kind="query_generation")

            queries = result_queries.get("queries", [])
            claim_decomp_raw = result_queries.get("claim")
            claim_decomp = self._normalize_claim_decomposition(claim_decomp_raw, fact=fact)
            self.last_query_meta = {"claim_decomposition": claim_decomp}
            
            if queries and len(queries) >= 1:
                # Filter out URLs and too-short queries
                valid = [q for q in queries if isinstance(q, str) and 5 < len(q) < 150 and "http" not in q.lower()]
                if valid:
                    # Pad with duplicates if needed
                    while len(valid) < len(languages_needed):
                        valid.append(valid[0])

                    # Enforce anchoring deterministically (defense-in-depth: the LLM can miss anchors).
                    missing_fields = [k for k in ("where", "when", "by_whom") if claim_decomp.get(k) is None]
                    anchored: list[str] = []
                    for (lang_code, _, _purpose), q in zip(languages_needed, valid):
                        q2 = q.strip()
                        anchors = [
                            claim_decomp.get("subject") or "",
                            self._action_anchor_for_lang(claim_decomp.get("action") or "", lang_code),
                            claim_decomp.get("object") or "",
                        ]
                        # Append missing-context probes when key attributes are missing.
                        probes = self._probe_phrases(lang_code=lang_code, missing=missing_fields)
                        # Keep the query compact; append only if the anchor isn't already present.
                        for a in anchors:
                            a = (a or "").strip()
                            if a and a.lower() not in q2.lower():
                                q2 = (q2 + " " + a).strip()
                        if probes:
                            # Ensure at least one probe is present when missing attributes exist.
                            probe = probes[0]
                            if probe and probe.lower() not in q2.lower():
                                q2 = (q2 + " " + probe).strip()
                        q2 = re.sub(r"\s+", " ", q2).strip()
                        if len(q2) > 150:
                            q2 = q2[:150].rstrip()
                        anchored.append(q2)
                    
                    # Log with language tags
                    for (lang_code, _, purpose), query in zip(languages_needed, anchored):
                        if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                            logger.debug("[M31] Generated %s query (%s): %s", purpose, lang_code, query[:80])
                    
                    if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
                        logger.debug("[GPT-5-Nano] ✓ Generated %d queries (anchored)", len(anchored[:len(languages_needed)]))
                        logger.debug("[M31] Claim decomposition: %s", claim_decomp)
                    return anchored[:len(languages_needed)]
            
            logger.warning("[GPT-5-Nano] No valid queries in response, using fallback")
            self.last_query_meta = {"claim_decomposition": self._normalize_claim_decomposition(None, fact=fact)}
            return self._smart_fallback(fact, lang, content_lang)
            
        except Exception as e:
            logger.warning("[GPT-5-Nano] ✗ Query generation error: %s", e)
            self.last_query_meta = {"claim_decomposition": self._normalize_claim_decomposition(None, fact=fact)}
            return self._smart_fallback(fact, lang, content_lang)
    
    def _smart_fallback(self, fact: str, lang: str = "en", content_lang: str = None) -> list[str]:
        """
        Smart fallback: Extract keywords from claim.
        M31: Returns 2-3 queries based on language overlap.
        """
        import re
        
        content_lang = content_lang or lang
        
        # Find numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', fact)
        
        # Find Latin words (potential English/scientific terms)
        latin_words = re.findall(r'\b[A-Za-z]{3,}\b', fact)
        
        keywords = []
        
        # Add acronyms (like NIST, NASA, GPS)
        acronyms = [w for w in latin_words if w.isupper() and len(w) >= 2]
        keywords.extend(acronyms[:3])
        
        # Add other Latin words
        other_latin = [w for w in latin_words if not w.isupper() and len(w) >= 4]
        keywords.extend(other_latin[:5])
        
        # Add key numbers
        keywords.extend(numbers[:2])
        
        # Build a deterministic minimal claim decomposition for anchoring (used when LLM query generation is unavailable).
        first_words = " ".join(re.findall(r"[^\s]+", (fact or "").strip())[:5]).strip()
        heur_claim = self._normalize_claim_decomposition(
            {
                "subject": first_words,
                "action": "claim",
                "object": " ".join([k for k in keywords if k])[:120] or first_words,
                "where": None,
                "when": None,
                "by_whom": None,
            },
            fact=fact,
        )
        self.last_query_meta = {"claim_decomposition": heur_claim}

        queries = []
        
        # Query 1: English keywords
        en_query = " ".join(keywords[:8]) if keywords else fact[:50].strip()
        queries.append(en_query)
        
        # Query 2: Content language (if != EN)
        if content_lang != "en":
            queries.append(fact[:100].strip())
        
        # Query 3: UI language (if different from both)
        if lang not in ["en", content_lang]:
            queries.append(fact[:100].strip())

        # Enforce anchors (subject/action/object + at least one probe) deterministically.
        anchored: list[str] = []
        lang_slots = ["en"]
        if content_lang != "en":
            lang_slots.append(content_lang)
        if lang not in ("en", content_lang):
            lang_slots.append(lang)
        missing = ["where", "when", "by_whom"]
        for code, q in zip(lang_slots, queries):
            q2 = (q or "").strip()
            anchors = [
                heur_claim.get("subject") or "",
                self._action_anchor_for_lang(heur_claim.get("action") or "", code),
                heur_claim.get("object") or "",
            ]
            probes = self._probe_phrases(lang_code=code, missing=missing)
            for a in anchors:
                a = (a or "").strip()
                if a and a.lower() not in q2.lower():
                    q2 = (q2 + " " + a).strip()
            if probes:
                p = probes[0]
                if p and p.lower() not in q2.lower():
                    q2 = (q2 + " " + p).strip()
            q2 = re.sub(r"\s+", " ", q2).strip()
            if len(q2) > 150:
                q2 = q2[:150].rstrip()
            anchored.append(q2)
        
        if ENGINE_DEBUG and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[M31] Smart fallback: %d queries for langs: en, %s, %s", len(queries), content_lang, lang)
        return anchored
