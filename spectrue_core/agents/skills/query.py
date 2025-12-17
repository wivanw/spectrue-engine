from .base_skill import BaseSkill, logger
from spectrue_core.verification.trusted_sources import AVAILABLE_TOPICS
from spectrue_core.constants import SUPPORTED_LANGUAGES
import re
from spectrue_core.agents.static_instructions import UNIVERSAL_METHODOLOGY_APPENDIX

class QuerySkill(BaseSkill):
    def __init__(self, config, llm_client):
        super().__init__(config, llm_client)
        self.last_query_meta = {}

    def _canonicalize_action(self, action: str | None) -> str:
        """
        Normalize action verbs to a small canonical set.
        This is used as an explicit anchor for query generation.
        """
        a = (action or "").strip().lower()
        if not a:
            return "claim"

        strong_map = {
            # English
            "ban": "ban", "banned": "ban", "prohibit": "prohibit", "prohibited": "prohibit",
            "forbid": "forbid", "forbidden": "forbid", "restrict": "restrict", "restricted": "restrict",
            # Ukrainian / Russian
            "заборона": "ban", "заборонили": "ban", "заборонено": "ban", "заборонити": "ban",
            "запрет": "ban", "запретили": "ban", "запрещено": "ban", "запретить": "ban",
            # German
            "verbot": "ban", "verboten": "ban", "untersagen": "ban", "untersagt": "ban",
            # Spanish
            "prohibir": "ban", "prohibido": "ban", "vetar": "ban", "vetado": "ban",
            "restringir": "restrict", "restringido": "restrict",
            # French
            "interdite": "ban", "interdire": "ban", "interdit": "ban", "bannir": "ban", "banni": "ban",
            # Japanese / Chinese (shared characters)
            "禁止": "ban", "制限": "restrict", "被禁": "ban", "限制": "restrict",
        }
        for k, v in strong_map.items():
            if k in a:
                return v

        if any(k in a for k in ("remove", "removed", "take down", "told to remove", "asked to remove")):
            return "remove"
        if any(k in a for k in ("arrest", "arrested", "detain", "detained")):
            return "arrest"
        if any(k in a for k in ("close", "closed", "shut down", "shutdown")):
            return "close"

        a = re.sub(r"\s+", " ", a).strip()
        return a[:32] if len(a) > 32 else a

    def _normalize_claim_decomposition(self, obj: dict | None, *, fact: str) -> dict:
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

    def _collapse_ws(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def _security_tokens(self) -> list[str]:
        return ["security", "guard", "guards", "sicherheit", "seguridad", "sécurité", "охорона", "безопасность", "警備", "保安"]

    def _contains_security_token(self, s: str) -> bool:
        t = (s or "").lower()
        for token in self._security_tokens():
            tok = token.lower()
            if any(ord(c) > 127 for c in tok):
                if tok in t:
                    return True
            else:
                if re.search(rf"\\b{re.escape(tok)}\\b", t, flags=re.IGNORECASE):
                    return True
        return False

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
                out = re.sub(rf"\\b{re.escape(tok)}\\b", "", out, flags=re.IGNORECASE)
        return self._collapse_ws(out)

    def _smart_fallback(self, fact: str, lang: str, content_lang: str) -> list[str]:
        """Simple fallback if LLM fails."""
        # Use simple splitting or just the fact itself (normalized)
        q = self._collapse_ws(fact)[:150]
        return [q, q]

    async def generate_search_queries(
        self,
        fact: str,
        context: str = "",
        lang: str = "en",
        content_lang: str = None,
        *,
        allow_short_llm: bool = False,
    ) -> list[str]:
        """
        Generate search queries using GPT-5 Nano.
        """
        target_lang_code = (content_lang or lang or "en").lower()
        
        # Determine language name for prompt
        target_lang_name = SUPPORTED_LANGUAGES.get(target_lang_code, "English")
        
        # Prepare content
        full_statement = fact if isinstance(fact, str) else str(fact)
        full_context = context if isinstance(context, str) and context else "None"
        
        # Input safety check (URL fallback)
        if (full_statement.startswith("http://") or full_statement.startswith("https://")) and len(full_statement.split()) < 3:
             # Logic from original agent to handle URLs directly
             # ...
             return [full_statement, full_statement] # Simplified for now

        topics_list_str = ", ".join(AVAILABLE_TOPICS)
        
        instructions = f"""You are a fact-checking search query generator.
Requirements:
- Output valid JSON (no markdown) with keys: "claim" (object), "queries" (list), and "topics" (list).
- "topics": select ALL matching topics from this list: [{topics_list_str}].
- Produce 2 queries:
  1) English: Pure factual query.
  2) {target_lang_name}: Pure factual query in local language.
- Queries MUST be specific.

You MUST respond in valid JSON.

{UNIVERSAL_METHODOLOGY_APPENDIX}
"""
        
        prompt = f"""Generate web search queries for fact-checking.

STATEMENT:
{full_statement}

CONTEXT:
{full_context}
"""
        
        # M56: Fix for OpenAI 400 "Response input messages must contain the word 'json'"
        # REQUIRED: The word "JSON" must appear in the INPUT message, not just system instructions.
        prompt += "\n\nReturn the result in JSON format."

        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions=instructions,
                reasoning_effort="low",
                cache_key=f"query_gen_v3_{target_lang_code}", # Stable prefix per language
                timeout=float(self.runtime.llm.nano_timeout_sec),
                trace_kind="query_generation",
            )
            
            raw_queries = result.get("queries", [])
            if not raw_queries:
                 raise ValueError("Empty queries")
            
            # Validation (word count, etc.) can be added here
            
            return raw_queries[:2]

        except Exception as e:
            logger.warning("[M48] Query generation failed: %s", e)
            # Fallback
            return [full_statement[:150], full_statement[:150]]
