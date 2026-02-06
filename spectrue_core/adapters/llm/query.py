# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import re

from .base_skill import BaseSkill, logger
from spectrue_core.tools.trusted_sources import AVAILABLE_TOPICS
from spectrue_core.constants import SUPPORTED_LANGUAGES
from spectrue_core.agents.static_instructions import UNIVERSAL_METHODOLOGY_APPENDIX
from spectrue_core.agents.llm_schemas import QUERY_GENERATION_SCHEMA
from spectrue_core.agents.llm_client import is_schema_failure
from spectrue_core.llm.model_registry import ModelID

FOLLOWUP_QUERY_TYPES = {
    "clarify_entity",
    "find_primary",
    "numeric_resolution",
    "temporal_disambiguation",
    "counterevidence_probe",
}


def _extract_snippets(sources: list[dict], max_snippets: int = 3) -> list[str]:
    snippets: list[str] = []
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        for key in ("quote", "snippet", "content"):
            text = src.get(key)
            if isinstance(text, str) and text.strip():
                snippets.append(text.strip())
                break
        if len(snippets) >= max_snippets:
            break
    return snippets


def _detect_query_type(claim_text: str, snippets: list[str]) -> str:
    claim_has_number = bool(re.search(r"\d", claim_text or ""))
    snippets_have_number = any(re.search(r"\d", s or "") for s in snippets)
    if claim_has_number and not snippets_have_number:
        return "numeric_resolution"

    claim_has_time = bool(re.search(r"\b(19|20)\d{2}\b", claim_text or ""))
    snippets_have_time = any(re.search(r"\b(19|20)\d{2}\b", s or "") for s in snippets)
    if claim_has_time and not snippets_have_time:
        return "temporal_disambiguation"

    if re.search(r"\b(announced|said|stated|according to|statement)\b", claim_text.lower() if claim_text else ""):
        return "find_primary"

    if any(re.search(r"\b(refute|deny|contradict|false)\b", s.lower() if s else "") for s in snippets):
        return "counterevidence_probe"

    return "clarify_entity"


def _build_query_from_snippet(snippet: str, query_type: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", snippet or "")
    base = " ".join(words[:12]).strip()
    if not base:
        return ""

    if query_type == "find_primary":
        return f"{base} official statement"
    if query_type == "numeric_resolution":
        return f"{base} exact number"
    if query_type == "temporal_disambiguation":
        return f"{base} timeline date"
    if query_type == "counterevidence_probe":
        return f"{base} contradicting evidence"
    return f"{base} source details"


def generate_followup_query_from_evidence(
    claim_text: str,
    sources: list[dict],
) -> dict[str, str] | None:
    """
    Build a follow-up query from evidence snippets, not the original article text.
    """
    snippets = _extract_snippets(sources)
    if not snippets:
        return None

    query_type = _detect_query_type(claim_text, snippets)
    if query_type not in FOLLOWUP_QUERY_TYPES:
        query_type = "clarify_entity"

    query = _build_query_from_snippet(snippets[0], query_type)
    if not query:
        return None

    return {
        "query": query,
        "query_type": query_type,
    }

class QuerySkill(BaseSkill):
    def __init__(self, config, llm_client):
        super().__init__(config, llm_client)
        self.last_query_meta = {}

    async def generate_search_queries(
        self,
        fact: str,
        context: str = "",
        lang: str = "en",
        content_lang: str = None,
        search_locale_plan: dict | None = None,
        time_sensitive: bool | None = None,
    ) -> list[str]:
        """
        Generate search queries using GPT-5 Nano.
        """
        primary_locale = (search_locale_plan or {}).get("primary") if search_locale_plan else None
        fallback_locales = (search_locale_plan or {}).get("fallback") if search_locale_plan else None
        target_lang_code = (primary_locale or content_lang or lang or "en").lower()
        fallback_lang_code = None
        if isinstance(fallback_locales, list) and fallback_locales:
            fallback_lang_code = fallback_locales[0]
        elif isinstance(fallback_locales, str):
            fallback_lang_code = fallback_locales

        # Determine language name for prompt
        target_lang_name = SUPPORTED_LANGUAGES.get(target_lang_code, "English")
        fallback_lang_name = SUPPORTED_LANGUAGES.get(fallback_lang_code, None) if fallback_lang_code else None

        # Prepare content
        full_statement = fact if isinstance(fact, str) else str(fact)
        full_context = context if isinstance(context, str) and context else "None"

        # Input safety check (URL fallback)
        if (full_statement.startswith("http://") or full_statement.startswith("https://")) and len(full_statement.split()) < 3:
             # Logic from original agent to handle URLs directly
             # ...
             return [full_statement, full_statement] # Simplified for now

        topics_list_str = ", ".join(AVAILABLE_TOPICS)

        recency_hint = ""
        if time_sensitive:
            recency_hint = "- If the claim is time-sensitive, bias queries toward recency (e.g., add year or \"latest\").\n"

        secondary_lang_line = ""
        if fallback_lang_name:
            secondary_lang_line = f"  2) {fallback_lang_name}: Pure factual query in fallback language.\n"
        else:
            secondary_lang_line = f"  2) {target_lang_name}: Pure factual query in local language.\n"

        instructions = f"""You are a fact-checking search query generator.
Requirements:
- Output valid JSON (no markdown) with keys: "claim" (object), "queries" (list), and "topics" (list).
- "claim": object with key "text" containing the TARGET_CLAIM.
- "topics": select ALL matching topics from this list: [{topics_list_str}].
- Produce 2 queries:
  1) English: Pure factual query.
{secondary_lang_line}{recency_hint}
- Queries MUST be specific.
- Generate search queries ONLY for the TARGET_CLAIM text.
- Ignore recipes, history, examples, or background unless explicitly referenced in the claim.

You MUST respond in valid JSON.

{UNIVERSAL_METHODOLOGY_APPENDIX}
"""

        prompt = f"""Generate web search queries for fact-checking.

TARGET_CLAIM:
{full_statement}

CONTEXT:
{full_context}
"""

        # Fix for OpenAI 400 "Response input messages must contain the word 'json'"
        # REQUIRED: The word "JSON" must appear in the INPUT message, not just system instructions.
        prompt += "\n\nReturn the result in JSON format."

        try:
            result = await self.llm_client.call_json(
                model=ModelID.NANO,
                input=prompt,
                instructions=instructions,
                response_schema=QUERY_GENERATION_SCHEMA,
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
            if is_schema_failure(e):
                raise
            # Fallback
            return [full_statement[:150], full_statement[:150]]
