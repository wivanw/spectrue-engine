# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Article Cleaner Skill - uses LLM Nano to extract clean article content.
"""

import logging
from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig

logger = logging.getLogger(__name__)

ARTICLE_CLEAN_PROMPT = """Extract ONLY the main article content from the text below.

REMOVE completely:
- Navigation, menus, headers
- "Read also", "See also", "Читайте також" sections
- Related articles, recommended content
- Advertisements, sidebars
- Social media links ("ТСН у соціальних мережах")
- News ticker ("Стрічка новин")
- Footer (address, phone, legal text)
- Cookie consent notices
- Copyright notices
- Author photos/bios
- Image captions like "Фото ілюстративне / © ..."
- "Читати публікацію повністю" links

KEEP only:
- Article title
- Article body text
- Important quotes and facts

Return ONLY the cleaned article. No explanations, no comments.

---
RAW TEXT:
{text}
---

CLEANED ARTICLE:"""


class ArticleCleanerSkill:
    """Uses LLM Nano to extract clean article content from raw page text."""
    
    def __init__(self, config: SpectrueConfig = None, llm_client: LLMClient = None):
        self.config = config
        self.runtime = (config.runtime if config else None) or EngineRuntimeConfig.load_from_env()
        self.llm_client = llm_client or LLMClient(
            openai_api_key=config.openai_api_key if config else None,
            default_timeout=float(self.runtime.llm.nano_timeout_sec),
        )
    
    async def clean_article(self, raw_text: str, *, max_input_chars: int = 12000) -> str | None:
        """
        Clean article text using LLM Nano.
        
        Args:
            raw_text: The raw page text (may include navigation, footer, etc.)
            max_input_chars: Maximum characters to send to LLM
            
        Returns:
            Cleaned article text, or None if cleaning fails
        """
        if not raw_text or len(raw_text) < 100:
            return raw_text
        
        # Truncate to avoid token limits
        truncated = raw_text[:max_input_chars]
        
        prompt = ARTICLE_CLEAN_PROMPT.format(text=truncated)
        
        try:

            
            result = await self.llm_client.call(
                model="gpt-5-nano",
                input=prompt,
                instructions="Extract only the main article content. Remove navigation, ads, related articles, footer.",
                json_output=False,
                cache_key=None,
                trace_kind="article_clean",
            )
            
            response = result.get("content", "") if isinstance(result, dict) else result
            

            
            if response and len(response.strip()) > 100:
                logger.info("[ArticleCleaner] Cleaned: %d -> %d chars", len(raw_text), len(response))
                return response.strip()
            else:
                logger.warning("[ArticleCleaner] LLM returned short response, using original")
                return raw_text
                
        except Exception as e:
            logger.warning("[ArticleCleaner] LLM cleaning failed: %s", e)
            return raw_text
