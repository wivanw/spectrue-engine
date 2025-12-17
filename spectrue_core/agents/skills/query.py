from .base_skill import BaseSkill, logger
from spectrue_core.verification.trusted_sources import AVAILABLE_TOPICS
from spectrue_core.constants import SUPPORTED_LANGUAGES
from spectrue_core.agents.static_instructions import UNIVERSAL_METHODOLOGY_APPENDIX

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
