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
import re
from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig
import asyncio
from spectrue_core.utils.text_chunking import CoverageSampler, TextChunk

logger = logging.getLogger(__name__)

ARTICLE_CLEAN_PROMPT = """Extract ONLY the main article content from the text below.

REMOVE completely:
- Navigation, menus, headers
- "Read also", "See also", "Читайте також" sections
- Related articles, recommended content
- Advertisements, sidebars
- Social media sharing buttons and "follow us" links
- News ticker ("Стрічка новин")
- Footer (address, phone, legal text)
- Cookie consent notices
- Copyright notices
- Author photos/bios
- Image captions like "Фото ілюстративне / © ..."
- "Читати публікацію повністю" links

KEEP (important!):
- Article title
- Article body text
- Important quotes and facts
- Source links and references (URLs to official sources, press releases, reports)
- Attribution links ("як повідомили в...", "за даними...", "according to...")

Return ONLY the cleaned article. No explanations, no comments.

---
RAW TEXT:
{text}
---

CLEANED ARTICLE:"""

ARTICLE_CLEAN_MARKDOWN_PROMPT = """Extract ONLY the main article content from the text below, formatted as clean Markdown.

REMOVE completely:
- Navigation, menus, headers
- "Read also", "See also", "Читайте також" sections
- Related articles, recommended content
- Advertisements, sidebars
- Social media sharing buttons and "follow us" links
- News ticker
- Footer (address, legal text, copyright)
- Cookie consent notices
- Author photos/bios
- Image captions like "Фото ілюстративне / © ..."
- "Читати публікацію повністю" links

KEEP and FORMAT:
- Article title (format as '# Title')
- Section headings (format as '##' or '###')
- Lists (format as '- Item')
- Article body text (separated by blank lines)
- Important quotes and facts
- Source links and references

Return ONLY the cleaned Markdown. No explanations.

---
RAW TEXT:
{text}
---

CLEANED MARKDOWN:"""


class ArticleCleanerSkill:
    """Uses LLM Nano to extract clean article content from raw page text."""
    
    # M73.5: Dynamic timeout constants
    BASE_TIMEOUT_SEC = 30.0    # Minimum timeout
    TIMEOUT_PER_1K_CHARS = 3.0  # Additional seconds per 1000 chars
    MAX_TIMEOUT_SEC = 90.0      # Maximum timeout cap
    
    def __init__(self, config: SpectrueConfig = None, llm_client: LLMClient = None):
        self.config = config
        self.runtime = (config.runtime if config else None) or EngineRuntimeConfig.load_from_env()
        self.llm_client = llm_client or LLMClient(
            openai_api_key=config.openai_api_key if config else None,
            default_timeout=float(self.runtime.llm.nano_timeout_sec),
        )
    
    def _calculate_timeout(self, text_len: int) -> float:
        """
        Calculate dynamic timeout based on text length.
        
        Formula: base + (chars / 1000) * per_1k_rate, capped at max.
        
        Examples:
            - 5000 chars → 30 + 15 = 45 sec
            - 12000 chars → 30 + 36 = 66 sec
            - 25000 chars → min(30 + 75, 90) = 90 sec
        """
        extra = (text_len / 1000) * self.TIMEOUT_PER_1K_CHARS
        timeout = self.BASE_TIMEOUT_SEC + extra
        return min(timeout, self.MAX_TIMEOUT_SEC)
    
    def _quick_clean_fallback(self, text: str) -> str:
        """
        Quick regex-based cleaning as fallback when LLM times out.
        Removes obvious navigation/junk patterns.
        """
        cleaned = text
        
        # Remove markdown images: ![alt](url) or !(/path)
        cleaned = re.sub(r'!\[.*?\]\([^)]+\)', '', cleaned)
        cleaned = re.sub(r'!\([^)]+\)', '', cleaned)
        
        # Remove navigation-style headers
        nav_patterns = [
            r'^#+\s*(What Are You Looking For|Popular Tags|Теги|Категорії|Menu|Navigation).*$',
            r'^#+\s*(Читайте також|Read also|See also|Дивіться також).*$',
            r'^#+\s*(Поділитись|Share|Follow us|Підписатись).*$',
            r'^\*\*?(Реклама|Advertisement|Спонсор)\*?\*?.*$',
        ]
        for pattern in nav_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove cookie/legal notices
        cleaned = re.sub(r'(?i)(cookie|cookies|gdpr|privacy policy|політика конфіденційності).*?\n', '', cleaned)
        
        # Remove excessive blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Remove lines that are just links or very short nav items
        lines = cleaned.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip very short lines that look like nav
            if len(stripped) < 20 and ('#' in stripped or '|' in stripped or stripped.startswith('-')):
                continue
            # Skip lines that are just a URL
            if re.match(r'^https?://\S+$', stripped):
                continue
            filtered_lines.append(line)
        
        cleaned = '\n'.join(filtered_lines).strip()
        
        logger.debug("[ArticleCleaner] Fallback regex clean: %d -> %d chars", len(text), len(cleaned))
        return cleaned
    
    async def clean_article(self, raw_text: str, *, max_input_chars: int = 12000) -> str | None:
        """
        Clean article text using LLM Nano.
        
        Args:
            raw_text: The raw page text (may include navigation, footer, etc.)
            max_input_chars: Maximum characters to send to LLM
            
        Returns:
            Cleaned article text, or fallback-cleaned text if LLM fails
        """
        if not raw_text or len(raw_text) < 100:
            return raw_text
            
        # M74: Coverage Chunking
        if self.runtime.features.coverage_chunking:
            merged, _ = await self.clean_article_chunked(raw_text)
            return merged
        
        # Truncate to avoid token limits
        truncated = raw_text[:max_input_chars]
        
        # M73.5: Calculate dynamic timeout based on input size
        dynamic_timeout = self._calculate_timeout(len(truncated))
        logger.debug("[ArticleCleaner] Input: %d chars, timeout: %.1f sec", len(truncated), dynamic_timeout)
        
        # M76: Select prompt based on feature flag
        if self.runtime.features.clean_md_output:
            prompt = ARTICLE_CLEAN_MARKDOWN_PROMPT.format(text=truncated)
            instr = "Extract the main article content as Markdown. Preserve headings (#, ##) and lists (-). Remove navigation, ads, footer."
        else:
            prompt = ARTICLE_CLEAN_PROMPT.format(text=truncated)
            instr = "Extract only the main article content. Remove navigation, ads, related articles, footer."
        
        try:
            result = await self.llm_client.call(
                model="gpt-5-nano",
                input=prompt,
                instructions=instr,
                json_output=False,
                cache_key=None,
                trace_kind="article_clean",
                timeout=dynamic_timeout,  # M73.5: Dynamic timeout
            )
            
            response = result.get("content", "") if isinstance(result, dict) else result
            
            if response and len(response.strip()) > 100:
                logger.debug("[ArticleCleaner] Cleaned: %d -> %d chars", len(raw_text), len(response))
                return response.strip()
            else:
                logger.warning("[ArticleCleaner] LLM returned short response, using fallback")
                return self._quick_clean_fallback(raw_text)
                
        except Exception as e:
            logger.warning("[ArticleCleaner] LLM cleaning failed: %s. Using fallback.", e)
            # M73.5: Use regex fallback instead of returning raw text
            return self._quick_clean_fallback(raw_text)

    async def clean_article_chunked(
        self, 
        raw_text: str, 
        *, 
        max_chunk_chars: int = 6000
    ) -> tuple[str, list[TextChunk]]:
        """
        M74: Clean article in chunks to prevent coverage loss.
        Returns (merged_clean_text, original_chunks).
        """
        sampler = CoverageSampler()
        chunks = sampler.chunk(raw_text, max_chunk_chars)
        
        if not chunks:
            return "", []
            
        logger.debug("[M74] Chunked cleaning: %d chunks for %d chars", len(chunks), len(raw_text))
        
        sem = asyncio.Semaphore(3)
        # Limit concurrency to 3
        
        async def _process_chunk(chunk: TextChunk) -> str:
            async with sem:
                timeout = self._calculate_timeout(len(chunk.text))
                # M76: Select prompt based on feature flag
                if self.runtime.features.clean_md_output:
                    prompt = ARTICLE_CLEAN_MARKDOWN_PROMPT.format(text=chunk.text)
                    instr = "Extract the main article content as Markdown. Preserve headings (#, ##) and lists (-). Remove navigation, ads, footer."
                else:
                    prompt = ARTICLE_CLEAN_PROMPT.format(text=chunk.text)
                    instr = "Extract only the main article content. Remove navigation, ads, related articles, footer."

                try:
                    result = await self.llm_client.call(
                        model="gpt-5-nano",
                        input=prompt,
                        instructions=instr,
                        json_output=False,
                        trace_kind="article_clean_chunk",
                        timeout=timeout,
                    )
                    content = result.get("content", "") if isinstance(result, dict) else result
                    # Lower threshold for chunks as they might be smaller
                    if content and len(content.strip()) > 50:
                        return content.strip()
                    return self._quick_clean_fallback(chunk.text)
                except Exception as e:
                    logger.warning("[M74] Chunk cleaning failed: %s. Using fallback.", e)
                    return self._quick_clean_fallback(chunk.text)

        cleaned_results = await asyncio.gather(*[_process_chunk(c) for c in chunks])
        merged = sampler.merge(list(cleaned_results))
        
        logger.debug("[M74] Merged length: %d chars", len(merged))
        return merged, chunks
