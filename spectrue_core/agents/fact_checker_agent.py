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

import json
import logging
import re
from openai import AsyncOpenAI
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.utils.security import sanitize_input
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig
import asyncio
from datetime import datetime
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.trusted_sources import AVAILABLE_TOPICS

logger = logging.getLogger(__name__)

class FactCheckerAgent:
    def __init__(self, config: SpectrueConfig = None):
        self.config = config
        self.runtime = (config.runtime if config else None) or EngineRuntimeConfig.load_from_env()
        api_key = config.openai_api_key if config else None
        
        self.client = AsyncOpenAI(api_key=api_key)
        
        self.timeout = float(self.runtime.llm.timeout_sec)
        self._sem = asyncio.Semaphore(int(self.runtime.llm.concurrency))
        # Keep lightweight nano query generation from queuing behind heavy final-analysis calls.
        self._sem_nano = asyncio.Semaphore(int(self.runtime.llm.nano_concurrency))
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
        # Never use the literal "claim" as an action anchor in queries.
        a_raw = (canonical_action or "").strip()
        if not a_raw or a_raw.lower() == "claim":
            return ""
        a = self._canonicalize_action(a_raw)
        if not a or a.lower() == "claim":
            return ""
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

    def _collapse_ws(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _security_tokens(self) -> list[str]:
        # Must match the prompt's security ban equivalents.
        return ["security", "guard", "guards", "sicherheit", "seguridad", "sécurité", "охорона", "безопасность", "警備", "保安"]

    def _contains_security_token(self, s: str) -> bool:
        t = (s or "").lower()
        for token in self._security_tokens():
            tok = token.lower()
            if any(ord(c) > 127 for c in tok):
                if tok in t:
                    return True
            else:
                if re.search(rf"\b{re.escape(tok)}\b", t, flags=re.IGNORECASE):
                    return True
        return False

    def _probe_lexicon(self, *, lang_code: str) -> list[str]:
        """
        Neutral probe phrases allowed for the given language code.
        This is used for deterministic validation only.
        """
        lc = (lang_code or "en").lower()
        return {
            "en": ["where", "when", "who said", "official statement"],
            "uk": ["де", "коли", "хто заявив", "офіційна заява"],
            "ru": ["где", "когда", "кто заявил", "официальное заявление"],
            "de": ["wo", "wann", "wer sagte", "offizielle erklärung"],
            "es": ["dónde", "cuándo", "quién dijo", "declaración oficial"],
            "fr": ["où", "quand", "qui a déclaré", "déclaration officielle"],
            "ja": ["どこ", "いつ", "誰が述べた", "公式声明"],
            "zh": ["哪里", "何时", "谁表示", "官方声明"],
        }.get(lc, ["where", "when", "who said", "official statement"])

    def _count_probe_occurrences(self, *, query: str, lang_code: str) -> int:
        """
        Return neutral probe phrase occurrences for the given query/lang.
        """
        q = self._collapse_ws(query).lower()
        neutral = self._probe_lexicon(lang_code=lang_code)

        def _occurrences(phrase: str) -> int:
            p = self._collapse_ws(phrase).lower()
            if not p:
                return 0
            if any(ord(c) > 127 for c in p):
                return q.count(p)
            pat = r"(?<!\w)" + re.escape(p).replace(r"\ ", r"\s+") + r"(?!\w)"
            return len(re.findall(pat, q, flags=re.IGNORECASE))

        return sum(_occurrences(p) for p in neutral)

    def _strip_security_tokens(self, *, query: str) -> str:
        q = self._collapse_ws(query)
        if not q:
            return q
        out = q
        for token in self._security_tokens():
            tok = token.lower()
            if any(ord(c) > 127 for c in tok):
                out = out.replace(token, "").replace(tok, "")
            else:
                out = re.sub(rf"\b{re.escape(tok)}\b", "", out, flags=re.IGNORECASE)
        return self._collapse_ws(out)

    def _ensure_query_anchors(self, *, query: str, claim: dict, lang_code: str) -> str:
        q = self._collapse_ws(query)
        subject = (claim.get("subject") or "").strip()
        obj_text = (claim.get("object") or "").strip()
        action_anchor = self._action_anchor_for_lang(claim.get("action") or "", lang_code)
        for a in (subject, action_anchor, obj_text):
            a = (a or "").strip()
            if not a:
                continue
            if a.lower() not in q.lower():
                q = (q + " " + a).strip()
        return self._collapse_ws(q)

    def build_strict_queries(self, claim_decomposition: dict | None, *, lang: str, content_lang: str | None) -> list[str]:
        """
        Deterministic "strict anchoring" queries used for a refined search pass.

        This intentionally avoids an additional LLM call; it is a real strategy change when relevance is low.
        """
        claim = self._normalize_claim_decomposition(claim_decomposition, fact="")
        content_lang = content_lang or lang

        def _build_for(code: str) -> str:
            action_anchor = self._action_anchor_for_lang(claim.get("action"), code)

            parts = [claim.get("subject") or "", action_anchor or "", claim.get("object") or ""]
            # Include available specifics to tighten the search.
            for k in ("where", "when", "by_whom"):
                v = claim.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            # Don't add probe phrases - they add noise and reduce relevance.

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
            "de": "WICHTIG: Erwähne in deiner Antwort keine internen Quellen-Tags wie [TRUSTED], [REL=…], [RAW].",
            "es": "IMPORTANTE: No menciones etiquetas de fuente internas como [TRUSTED], [REL=…], [RAW] en tu respuesta.",
            "fr": "IMPORTANT : Ne mentionne pas les balises de source internes comme [TRUSTED], [REL=…], [RAW] dans ta réponse.",
            "ja": "重要：出力に [TRUSTED]、[REL=…]、[RAW] のような内部ソースタグを含めないでください。",
            "zh": "重要：请勿在输出中提及 [TRUSTED]、[REL=…]、[RAW] 等内部来源标签。",
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

        if self.runtime.debug.log_prompts:
            logger.debug("--- Prompt for %s (%s) ---\n%s\n--- End Prompt ---", gpt_model, lang, prompt)
        elif self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Built prompt for %s (%s): %d chars", gpt_model, lang, len(prompt))
        
        messages = [{"role": "user", "content": prompt}]

        def _max_output_tokens_for_mode(mode: str) -> int:
            if mode == "lite":
                return int(self.runtime.llm.max_output_tokens_lite)
            if mode == "deep":
                return int(self.runtime.llm.max_output_tokens_deep)
            return int(self.runtime.llm.max_output_tokens_general)

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

        last_exception = None
        result = {}

        for attempt in range(2):  # Try once, then retry once
            try:
                if self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
                    logger.debug("[GPT] Calling %s (attempt %d/2) with %d chars prompt...", gpt_model, attempt + 1, len(prompt))
                async with self._sem:
                    response = await self.client.chat.completions.create(**api_params)
                
                raw_content = response.choices[0].message.content
                Trace.event(
                    "llm.response",
                    {
                        "kind": "analysis",
                        "model": gpt_model,
                        "attempt": attempt + 1,
                        "content_chars": len(raw_content or ""),
                        "content": raw_content,
                    },
                )
                
                if not raw_content or not raw_content.strip():
                    raise ValueError("Empty response from LLM")

                if self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
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
                # Success - break loop
                break
            except Exception as e:
                last_exception = e
                logger.warning("[GPT] Attempt %d failed for %s: %s", attempt + 1, gpt_model, e)
                if attempt == 1: # Last attempt failed
                    logger.exception("[GPT] ✗ Error calling %s after retries: %s", gpt_model, e)
                    Trace.event("llm.error", {"kind": "analysis", "model": gpt_model, "error": str(e)})
                    return {
                        "verified_score": 0.5,
                        "context_score": 0.5,
                        "danger_score": 0.0,
                        "style_score": 0.5,
                        "confidence_score": 0.2,
                        "rationale_key": "errors.agent_call_failed"
                    }
                continue # Retry

        # Post-processing (outside loop)
        if self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[GPT] ✓ JSON parsed successfully. Keys: %s", list(result.keys()))
        
        # Handle Chain-of-Thought filtering (support both keys)
        thought_process = result.pop("thought_process", None) or result.pop("_thought_process", None)
        if thought_process and self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
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
        
        if self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[GPT] Scores: V=%.2f, D=%.2f, Conf=%.2f",
                float(result.get("verified_score", 0) or 0),
                float(result.get("danger_score", 0) or 0),
                float(result.get("confidence_score", 0) or 0),
            )
        return result

    async def generate_search_queries(
        self,
        fact: str,
        context: str = "",
        lang: str = "en",
        content_lang: str = None,  # M31: Content language from detection
        *,
        allow_short_llm: bool = False,
    ) -> list[str]:
        """
        Generate search queries using GPT-5 Nano.

        Updated: always send the full STATEMENT verbatim; generate exactly 2 queries (EN, then UK).
        """
        import json

        # Determine target language dynamically
        target_lang_code = (content_lang or lang or "en").lower()
        lang_names = {
            "en": "English",
            "uk": "Ukrainian",
            "ru": "Russian",
            "de": "German",
            "es": "Spanish",
            "fr": "French",
            "ja": "Japanese",
            "zh": "Chinese",
        }
        target_lang_name = lang_names.get(target_lang_code, "English")

        # Always generate exactly 2 queries: [EN global, Target content].
        # If target is EN, we just ask for English twice (or similar), effectively getting 2 variations.
        languages_needed = [("en", "English", "global"), (target_lang_code, target_lang_name, "content")]

        if self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[M31] Generating queries for languages: %s", [l[0] for l in languages_needed])
        
        # No manual truncation/compression: the nano prompt receives the full statement/context verbatim.
        full_statement = fact if isinstance(fact, str) else str(fact)
        if context is None:
            full_context_or_None = "None"
        else:
            full_context_or_None = context if isinstance(context, str) else str(context)
            if not full_context_or_None:
                full_context_or_None = "None"
        
        # Build prompt dynamically
        language_instructions = f"1) English\n2) {target_lang_name} (if translation fails, repeat English)"

        def _sanitize_prompt_for_trace(prompt_text: str) -> dict:
            """Sanitize long prompts for trace (len/sha256/head/tail format)."""
            import hashlib
            if len(prompt_text) <= 500:
                return prompt_text
            head_len = min(200, len(prompt_text) // 2)
            tail_len = min(200, len(prompt_text) // 2)
            return {
                "len": len(prompt_text),
                "sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                "head": prompt_text[:head_len],
                "tail": prompt_text[-tail_len:] if tail_len else "",
            }

        async def _call_query_llm(prompt_text: str, *, kind: str) -> dict:
            Trace.event(
                "llm.prompt",
                {
                    "kind": kind,
                    "model": "gpt-5-nano",
                    "lang": lang,
                    "content_lang": content_lang,
                    "prompt_chars": len(prompt_text),
                    "prompt": _sanitize_prompt_for_trace(prompt_text),
                },
            )
            # Query generation should be fast and bounded.
            # Use a dedicated semaphore so nano doesn't wait behind analysis calls.
            api_params = {
                "model": "gpt-5-nano",
                "messages": [{"role": "user", "content": prompt_text}],
                "response_format": {"type": "json_object"},
                "timeout": float(self.runtime.llm.nano_timeout_sec),
            }
            # Query generation uses JSON format - no need for token limit.
            # GPT-5 reasoning models need these parameters for efficiency.
            if "gpt-5" in api_params["model"] or api_params["model"].startswith("o1-") or "o1" in api_params["model"]:
                api_params["reasoning_effort"] = "low"
                api_params["verbosity"] = "low"

            async with self._sem_nano:
                response = await self.client.chat.completions.create(**api_params)
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
            # Diagnostic: log response shape when content is empty
            if not isinstance(raw, str) or not raw.strip():
                finish_reason = getattr(response.choices[0], "finish_reason", None) if response.choices else None
                usage_obj = getattr(response, "usage", None)
                usage_dict = None
                if usage_obj:
                    try:
                        usage_dict = {
                            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                            "total_tokens": getattr(usage_obj, "total_tokens", None),
                        }
                    except Exception:
                        usage_dict = str(usage_obj)
                Trace.event(
                    "llm.empty_response",
                    {
                        "kind": kind,
                        "model": "gpt-5-nano",
                        "finish_reason": finish_reason,
                        "usage": usage_dict,
                        "choices_count": len(response.choices) if response.choices else 0,
                    },
                )
                logger.warning(
                    "[GPT-5-Nano] Empty response for %s (finish_reason=%s, usage=%s)",
                    kind, finish_reason, usage_dict
                )
                return {}
            try:
                result = json.loads(raw)
            except Exception:
                return {}
            Trace.event(
                "llm.parsed",
                {
                    "kind": kind,
                    "model": "gpt-5-nano",
                    "keys": list(result.keys()),
                },
            )
            return result

        # Build topics list string for prompt
        topics_list_str = ", ".join(AVAILABLE_TOPICS)

        prompt_queries = f"""Generate web search queries for fact-checking.

Requirements:
- Output valid JSON (no markdown) with keys: "claim", "queries", and "topics".
- "claim" MUST have exactly: subject, action, object, where, when, by_whom (strings or null).
  - "where": geographic location ONLY (country, city, region). Use null for space/sky events or if not specified.
  - "when": specific date/period if mentioned, otherwise null.
  - "by_whom": source/author/organization that made the claim (e.g. "Daily Galaxy", "NASA"). Extract from text if mentioned.
  - Include verifiable specifics in action/object: distances, magnitudes, scientific terms (e.g. "white dwarf", "supernova", "10,000 light-years").
- Produce EXACTLY 2 queries in order: 1) English, 2) Ukrainian. If Ukrainian is impossible, repeat the English query.
- Each query: concise (6–14 words), no URLs/hashtags/quotes, neutral wording. Do NOT add probe phrases or domain-specific markers.
- Do NOT introduce security/guard terms unless they already appear in STATEMENT or CONTEXT.
- "topics": select ALL matching topics from this list: [{topics_list_str}]. Empty list if none match.

STATEMENT (verbatim):
{full_statement}

CONTEXT (optional, verbatim or "None"):
{full_context_or_None}
"""

        prompt_queries_retry = f"""Return JSON with "claim" + "queries" + "topics".
- queries: 2 items (English then Ukrainian; repeat EN if needed), 6–14 words, neutral.
- topics: select from [{topics_list_str}] or empty list.
STATEMENT:
{full_statement}
CONTEXT:
{full_context_or_None}
"""

        try:
            if self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
                logger.debug("[GPT-5-Nano] Generating %d queries for: %s...", len(languages_needed), (full_statement or "")[:80])

            result_queries = await _call_query_llm(prompt_queries, kind="query_generation")
            if not result_queries:
                # One retry without CONTEXT if nano returned empty or invalid JSON.
                result_queries = await _call_query_llm(prompt_queries_retry, kind="query_generation.retry")
            if not isinstance(result_queries, dict) or not result_queries:
                raise ValueError("query_generation: empty/invalid response")

            # Contract validation: top-level keys.
            expected_keys = {"claim", "queries", "topics"}
            actual_keys = set(result_queries.keys())
            # Accept with or without topics for backward compatibility
            if "claim" not in actual_keys or "queries" not in actual_keys:
                raise ValueError("query_generation: missing claim or queries keys")

            claim_decomp_raw = result_queries.get("claim")
            if not isinstance(claim_decomp_raw, dict) or set(claim_decomp_raw.keys()) != {
                "subject",
                "action",
                "object",
                "where",
                "when",
                "by_whom",
            }:
                raise ValueError("query_generation: invalid claim keys")

            claim_decomp = self._normalize_claim_decomposition(claim_decomp_raw, fact=fact)
            
            # M45: Extract topics from LLM response
            raw_topics = result_queries.get("topics", [])
            if not isinstance(raw_topics, list):
                raw_topics = []
            # Validate topics against AVAILABLE_TOPICS
            valid_topics = [t for t in raw_topics if isinstance(t, str) and t.lower().strip() in [at.lower() for at in AVAILABLE_TOPICS]]
            
            self.last_query_meta = {"claim_decomposition": claim_decomp, "topics": valid_topics}

            missing_fields = [k for k in ("where", "when", "by_whom") if claim_decomp.get(k) is None]
            statement_has_security = self._contains_security_token(full_statement or "")

            raw_queries = result_queries.get("queries")
            if not isinstance(raw_queries, list):
                raw_queries = []
            raw_queries = [q for q in raw_queries if isinstance(q, str) and q.strip()]

            # Enforce exactly 2 queries, EN then UK; repeat EN if missing.
            if len(raw_queries) < 1:
                raise ValueError("query_generation: empty queries")
            if len(raw_queries) < 2:
                raw_queries.append(raw_queries[0])
            raw_queries = raw_queries[:2]

            anchored: list[str] = []
            for (lang_code, _, _purpose), q in zip(languages_needed, raw_queries):
                q2 = self._collapse_ws(q)

                # Security ban (generic): remove security tokens unless explicitly present in the statement.
                if not statement_has_security and self._contains_security_token(q2):
                    q2 = self._strip_security_tokens(query=q2)

                # Enforce length rule: 6–14 words.
                word_count = len(q2.split())
                if word_count < 6 or word_count > 14:
                    raise ValueError("query_generation: invalid word count")

                # Enforce max query length 150 chars (safety, not input truncation).
                q2 = q2[:150] if len(q2) > 150 else q2
                anchored.append(q2)

            if self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
                for (lang_code, _, purpose), query in zip(languages_needed, anchored):
                    logger.debug("[M31] Generated %s query (%s): %s", purpose, lang_code, query[:80])
                logger.debug("[GPT-5-Nano] ✓ Generated %d queries (validated)", len(anchored))
                logger.debug("[M31] Claim decomposition: %s", claim_decomp)

            return anchored

        except Exception as e:
            logger.warning("[GPT-5-Nano] ✗ Query generation error: %s", e)
            self.last_query_meta = {"claim_decomposition": self._normalize_claim_decomposition(None, fact=fact)}
            return self._smart_fallback(fact, lang, content_lang)
    
    def _smart_fallback(self, fact: str, lang: str = "en", content_lang: str = None) -> list[str]:
        """
        Deterministic fallback: extract key terms and build 2 concise queries (EN, UK).
        Avoids truncating words mid-way.
        """
        import re
        content_lang = content_lang or lang

        raw_fact = (fact or "").strip()
        if not raw_fact:
            self.last_query_meta = {"claim_decomposition": self._normalize_claim_decomposition(None, fact="")}
            return ["", ""]

        # Extract proper nouns (Latin: capitalized words)
        proper_nouns_latin = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', raw_fact)
        # Extract proper nouns (Cyrillic: capitalized words)
        proper_nouns_cyrillic = re.findall(r'\b[А-ЯІЇЄҐ][а-яіїєґ]+(?:\s+[А-ЯІЇЄҐ][а-яіїєґ]+)*\b', raw_fact)
        # Extract numbers (dates, amounts, distances)
        numbers = re.findall(r'\b\d[\d\s,.]*\d?\b', raw_fact)
        # Clean numbers (remove internal spaces)
        numbers = [re.sub(r'\s+', '', n).strip() for n in numbers if n.strip()]

        # Build key terms list (prioritize proper nouns, then numbers)
        key_terms = []
        seen = set()
        for term in proper_nouns_latin[:4] + proper_nouns_cyrillic[:3] + numbers[:2]:
            term_clean = term.strip()
            if term_clean and term_clean.lower() not in seen:
                seen.add(term_clean.lower())
                key_terms.append(term_clean)

        # If no proper nouns found, use first sentence (max 10 words)
        if not key_terms:
            first_sentence = re.split(r'[.!?\n]', raw_fact)[0].strip()
            words = first_sentence.split()[:10]
            key_terms = [" ".join(words)] if words else [raw_fact[:100]]

        base_query = " ".join(key_terms).strip()
        base_query = re.sub(r'\s+', ' ', base_query)

        # Ensure word boundary (don't cut mid-word) - max 150 chars
        if len(base_query) > 150:
            base_query = base_query[:150].rsplit(' ', 1)[0]

        heur_claim = {
            "subject": key_terms[0] if key_terms else "",
            "action": self._canonicalize_action(raw_fact) or "claim",
            "object": " ".join(key_terms[1:3]) if len(key_terms) > 1 else base_query[:120],
            "where": None,
            "when": None,
            "by_whom": None,
        }
        self.last_query_meta = {"claim_decomposition": heur_claim}

        # EN and UK queries are the same in fallback (key terms are language-agnostic)
        en_query = base_query
        uk_query = base_query

        if self.runtime.debug.engine_debug and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[M44] Smart fallback: EN='%s', UK='%s'", en_query[:80], uk_query[:80])
        return [en_query, uk_query]