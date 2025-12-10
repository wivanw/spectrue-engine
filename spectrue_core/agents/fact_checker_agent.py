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
from openai import AsyncOpenAI
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.utils.security import sanitize_input
from spectrue_core.config import SpectrueConfig
import asyncio
from datetime import datetime


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

    async def analyze(
        self, 
        fact: str, 
        context: str, 
        gpt_model: str, 
        lang: str, 
        analysis_mode: str = "general",
        search_mode: str = "basic"
    ) -> dict:
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
        # M39: Pass search_mode to prompt for smart recommendations
        prompt = prompt_template.format(
            fact=safe_fact, 
            context=safe_context, 
            date=current_date,
            search_mode=search_mode
        )
        
        # Debug: Log the prompt
        print(f"--- Prompt for {gpt_model} ({lang}) ---")
        print(prompt)
        print("--- End Prompt ---")
        
        messages = [{"role": "user", "content": prompt}]

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
        
        # Optimize speed for reasoning models
        if "gpt-5" in gpt_model or "o1" in gpt_model:
            api_params["reasoning_effort"] = "low"
            # GPT-5 and o1 models don't support temperature parameter
            use_temperature = False
        
        # Add temperature only if supported by the model
        if use_temperature:
            api_params["temperature"] = 0  # Deterministic for fact-checking

        try:
            print(f"[GPT] Calling {gpt_model} with {len(prompt)} chars prompt...")
            async with self._sem:
                response = await self.client.chat.completions.create(**api_params)
            
            raw_content = response.choices[0].message.content
            print(f"[GPT] Got response: {len(raw_content)} chars")
            print(f"[GPT] Response preview: {raw_content[:300]}...")
            
            # Parse the JSON response
            result = json.loads(raw_content)
            print(f"[GPT] ✓ JSON parsed successfully. Keys: {list(result.keys())}")
            
            # Handle Chain-of-Thought filtering (support both keys)
            thought_process = result.pop("thought_process", None) or result.pop("_thought_process", None)
            if thought_process:
                print(f"[Aletheia-X Thought]: {thought_process}")
            
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
            
            print(f"[GPT] Scores: V={result.get('verified_score', 0):.2f}, D={result.get('danger_score', 0):.2f}, Conf={result.get('confidence_score', 0):.2f}")
            return result
        except json.JSONDecodeError as e:
            print(f"[GPT] ✗ JSON parse error: {e}")
            print(f"[GPT] Raw content was: {raw_content[:500] if 'raw_content' in locals() else 'N/A'}")
            return {
                "verified_score": 0.5,
                "context_score": 0.5,
                "danger_score": 0.0,
                "style_score": 0.5,
                "confidence_score": 0.2,
                "rationale_key": "errors.agent_call_failed"
            }
        except Exception as e:
            print(f"[GPT] ✗ Error calling {gpt_model}: {e}")
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
        
        print(f"[M31] Generating {len(languages_needed)} queries for languages: {[l[0] for l in languages_needed]}")
        
        # Clean input
        clean_fact = re.sub(r'http\S+', '', fact[:800]).replace("\n", " ").strip()
        clean_context = context[:500].replace("\n", " ").strip() if context else ""
        
        # Build prompt for multi-language generation
        language_instructions = "\n".join([
            f"{i+1}. {name} ({code}) - {purpose} search"
            for i, (code, name, purpose) in enumerate(languages_needed)
        ])
        
        prompt = f"""Generate exactly {len(languages_needed)} search queries to fact-check this claim.

RULES:
1. Extract KEY entities: names, numbers, organizations, dates
2. Keep queries concise (5-10 words each)
3. Focus on the MAIN claim only
4. Each query MUST be in its specified language

LANGUAGES NEEDED:
{language_instructions}

OUTPUT FORMAT (JSON only):
{{"queries": [{", ".join([f'"{lang[1].lower()} query"' for lang in languages_needed])}]}}

CLAIM: {clean_fact}
CONTEXT: {clean_context if clean_context else "None"}"""

        try:
            print(f"[GPT-5-Nano] Generating {len(languages_needed)} queries for: {clean_fact[:80]}...")
            async with self._sem:
                response = await self.client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    # Note: GPT-5 models don't support temperature parameter
                    timeout=30
                )
            
            raw = response.choices[0].message.content
            print(f"[GPT-5-Nano] Raw response: {raw[:200]}...")
            
            result = json.loads(raw)
            queries = result.get("queries", [])
            
            if queries and len(queries) >= 1:
                # Filter out URLs and too-short queries
                valid = [q for q in queries if isinstance(q, str) and 5 < len(q) < 150 and "http" not in q.lower()]
                if valid:
                    # Pad with duplicates if needed
                    while len(valid) < len(languages_needed):
                        valid.append(valid[0])
                    
                    # Log with language tags
                    for (lang_code, _, purpose), query in zip(languages_needed, valid):
                        print(f"[M31] Generated {purpose} query ({lang_code}): {query[:80]}")
                    
                    print(f"[GPT-5-Nano] ✓ Generated {len(valid[:len(languages_needed)])} queries")
                    return valid[:len(languages_needed)]
            
            print("[GPT-5-Nano] No valid queries in response, using fallback")
            return self._smart_fallback(fact, lang, content_lang)
            
        except Exception as e:
            print(f"[GPT-5-Nano] ✗ Query generation error: {e}")
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
        
        print(f"[M31] Smart fallback: {len(queries)} queries for langs: en, {content_lang}, {lang}")
        return queries