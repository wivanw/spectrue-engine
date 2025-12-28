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

"""
Text analysis utilities for HTML parsing and sentence segmentation.

This module provides the TextAnalyzer class for:
- Parsing HTML content into clean text using trafilatura
- Segmenting text into sentences using spaCy
- Extracting metadata (title, authors, publish date)
"""

import logging
import asyncio
import hashlib
from typing import List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
# newspaper3k replaced with trafilatura
import trafilatura

from spectrue_core.analysis.content_budgeter import ContentBudgeter, TrimResult
from spectrue_core.runtime_config import ContentBudgetConfig
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class ParsedText:
    """Clean text output from HTML parsing."""
    text: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    publish_date: Optional[datetime] = None
    raw_text: Optional[str] = None
    raw_len: Optional[int] = None
    cleaned_len: Optional[int] = None
    raw_sha256: Optional[str] = None
    cleaned_sha256: Optional[str] = None
    selection_meta: Optional[List[dict]] = None
    blocks_stats: Optional[dict] = None


@dataclass
class SentenceSegment:
    """Individual sentence extracted by spaCy."""
    text: str
    start_char: int
    end_char: int
    index: int


class TextAnalyzer:
    """
    Analyzes text by parsing HTML and segmenting sentences.
    
    Uses trafilatura for HTML parsing and spaCy for sentence segmentation.
    Lazy-loads spaCy model to avoid startup delays.
    Supports 8 languages with lightweight models.
    """
    
    # Mapping of language codes to lightweight spaCy models
    SPACY_MODELS = {
        "en": "en_core_web_sm",      # English
        "uk": "uk_core_news_sm",     # Ukrainian
        "ru": "ru_core_news_sm",     # Russian
        "de": "de_core_news_sm",     # German
        "es": "es_core_news_sm",     # Spanish
        "fr": "fr_core_news_sm",     # French
        "ja": "ja_core_news_sm",     # Japanese
        "zh": "zh_core_web_sm",      # Chinese
    }
    
    def __init__(self, verifier=None, config=None):
        """Initialize TextAnalyzer with lazy-loaded models."""
        self._nlp_cache = {}  # Cache loaded models by language
        self.verifier = verifier
        self.config = config or {}

    def _get_budget_config(self) -> ContentBudgetConfig:
        runtime_cfg = getattr(self.config, "runtime", None)
        cfg = getattr(runtime_cfg, "content_budget", None)
        if isinstance(cfg, ContentBudgetConfig):
            return cfg
        return ContentBudgetConfig()
        
    def _get_nlp(self, language: str):
        """
        Lazy-load spaCy model for the specified language.
        
        Args:
            language: Language code (en, uk, ru, de, es, fr, ja, zh)
            
        Returns:
            Loaded spaCy model
            
        Raises:
            ValueError: If language is not supported
        """
        import spacy

        if language not in self.SPACY_MODELS:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.SPACY_MODELS.keys())}")
        
        # Return cached model if available
        if language in self._nlp_cache:
            return self._nlp_cache[language]
        
        model_name = self.SPACY_MODELS[language]
        
        try:
            nlp = spacy.load(model_name)
            self._nlp_cache[language] = nlp
            return nlp
        except OSError:
            # Model not found, try to download
            logger.warning("spaCy model '%s' not found. Attempting download...", model_name)
            import subprocess
            import sys
            try:
                subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
                nlp = spacy.load(model_name)
                self._nlp_cache[language] = nlp
                return nlp
            except Exception as e:
                raise ValueError(f"Failed to load spaCy model for {language}: {e}")
    
    def _preserve_links_in_html(self, html_content: str) -> str:
        """
        M29: Preserve URLs by appending them to anchor text before parsing.
        
        Converts: <a href="https://space.com/article">Space.com</a>
        To: Space.com (https://space.com/article)
        
        This helps the LLM see source citations during fact-checking.
        """
        import re
        
        def replace_link(match):
            full_tag = match.group(0)
            href_match = re.search(r'href=["\']([^"\']+)["\']', full_tag, re.IGNORECASE)
            
            # Extract the text content between <a> and </a>
            text_match = re.search(r'>([^<]*)</a>', full_tag, re.IGNORECASE)
            
            if not href_match or not text_match:
                return full_tag
            
            href = href_match.group(1).strip()
            text = text_match.group(1).strip()
            
            # Only preserve valid http/https links, skip internal anchors, javascript, mailto
            if not href or not text:
                return full_tag
            if not re.match(r'^https?://', href, re.IGNORECASE):
                return full_tag
            # Skip if href is already in the text (no duplication needed)
            if href in text:
                return full_tag
            
            # Return text with URL appended
            return f'{text} ({href})'
        
        # Match <a ...>text</a> patterns
        pattern = r'<a\s+[^>]*href=[^>]+>[^<]*</a>'
        return re.sub(pattern, replace_link, html_content, flags=re.IGNORECASE)
    
    def parse_html(self, html_content: str, language: Optional[str] = None) -> ParsedText:
        """
        Parse HTML content into clean text using trafilatura.
        
        Args:
            html_content: Raw HTML string
            
        Returns:
            ParsedText object with extracted text and metadata
            
        Raises:
            ValueError: If HTML is empty or parsing fails
        """
        if not html_content or not html_content.strip():
            raise ValueError("HTML content cannot be empty")
        
        # M29: Preserve links before parsing
        html_with_links = self._preserve_links_in_html(html_content)
        
        budget_cfg = self._get_budget_config()
        budget_result: Optional[TrimResult] = None

        try:
            # Use trafilatura for extraction
            text = trafilatura.extract(
                html_with_links,
                include_comments=False,
                include_tables=True,
                include_links=True,
                include_images=False,
                no_fallback=False
            )
            
            # Extract metadata separately
            metadata = trafilatura.extract_metadata(html_with_links)
            
            if not text or not text.strip():
                raise ValueError("Parsed text is empty")

            extracted_text = text.strip()
            raw_len = len(extracted_text)

            if raw_len > int(budget_cfg.absolute_guardrail_chars):
                Trace.event(
                    "content_budgeter.guardrail",
                    {"raw_len": raw_len, "absolute_guardrail_chars": int(budget_cfg.absolute_guardrail_chars)},
                )
                raise ValueError("HTML content too large to process safely")

            if raw_len > int(budget_cfg.max_clean_text_chars_default):
                try:
                    budget_result = ContentBudgeter(budget_cfg).trim(extracted_text)
                except ValueError:
                    # Guardrail already traced above; re-raise
                    raise

            cleaned_text = budget_result.trimmed_text if budget_result else extracted_text
            cleaned_len = len(cleaned_text)
            raw_sha = budget_result.raw_sha256 if budget_result else hashlib.sha256(
                extracted_text.encode("utf-8")
            ).hexdigest()
            cleaned_sha = budget_result.trimmed_sha256 if budget_result else hashlib.sha256(
                cleaned_text.encode("utf-8")
            ).hexdigest()

            if budget_result:
                Trace.event("content_budgeter.blocks", budget_result.trace_blocks_payload())
                Trace.event(
                    "content_budgeter.selection",
                    budget_result.trace_selection_payload(getattr(budget_cfg, "trace_top_blocks", 8)),
                )

            Trace.event(
                "analysis.input_summary",
                {
                    "source": "html",
                    "raw_len": raw_len,
                    "cleaned_len": cleaned_len,
                    "raw_sha256": raw_sha,
                    "cleaned_sha256": cleaned_sha,
                    "budget_applied": bool(budget_result),
                },
            )
            
            title = None
            authors = None
            publish_date = None
            
            if metadata:
                title = getattr(metadata, 'title', None)
                author = getattr(metadata, 'author', None)
                if author:
                    authors = [author] if isinstance(author, str) else list(author)
                date_str = getattr(metadata, 'date', None)
                if date_str:
                    try:
                        publish_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        pass
            
            return ParsedText(
                text=cleaned_text,
                title=title,
                authors=authors,
                publish_date=publish_date,
                raw_text=extracted_text,
                raw_len=raw_len,
                cleaned_len=cleaned_len,
                raw_sha256=raw_sha,
                cleaned_sha256=cleaned_sha,
                selection_meta=budget_result.selection_meta if budget_result else None,
                blocks_stats=budget_result.blocks_stats if budget_result else None,
            )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to parse HTML: {e}")

    def fetch_url(self, url: str, language: str = "en") -> ParsedText:
        """
        Fetch and parse content from a URL using trafilatura.
        
        Args:
            url: URL to fetch
            language: Language code
            
        Returns:
            ParsedText object
            
        Raises:
            ValueError: If fetching or parsing fails
        """
        if not url:
            raise ValueError("URL cannot be empty")
        
        try:
            # Fetch using trafilatura
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ValueError(f"Failed to download URL: {url}")
            
            # Parse the downloaded content
            return self.parse_html(downloaded)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to fetch/parse URL: {e}")

    def segment_sentences(self, text: str, language: str = "en") -> List[SentenceSegment]:
        """
        Segment text into sentences using spaCy.
        
        Args:
            text: Input text to segment
            language: Language code for model selection (default: en)
            
        Returns:
            List of SentenceSegment objects
            
        Raises:
            ValueError: If text is empty or language unsupported
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Get language-specific spaCy model
        nlp = self._get_nlp(language)
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract sentences
        segments = []
        for idx, sent in enumerate(doc.sents):
            segments.append(SentenceSegment(
                text=sent.text.strip(),
                start_char=sent.start_char,
                end_char=sent.end_char,
                index=idx
            ))
        
        return segments
    
    def parse_and_segment(self, html_content: str, language: str = "en") -> tuple[ParsedText, List[SentenceSegment]]:
        """
        Convenience method to parse HTML and segment sentences in one call.
        
        Args:
            html_content: Raw HTML string
            language: Language code for parsing and segmentation
            
        Returns:
            Tuple of (ParsedText, List[SentenceSegment])
        """
        parsed = self.parse_html(html_content)
        segments = self.segment_sentences(parsed.text, language)
        return parsed, segments

    def get_sentences(self, text: str, lang: str) -> List[str]:
        """Wrapper for segment_sentences to return list of strings (compatibility)."""
        segments = self.segment_sentences(text, lang)
        return [s.text for s in segments]

    async def analyze_text(self, sentences: List[str], search_type: str, gpt_model: str, lang: str, 
                          progress_callback: Optional[Callable] = None, analysis_mode: str = "general",
                          global_context: str = None, global_sources: list = None,
                          content_lang: str = None) -> dict:  # M31: Added content_lang
        """
        Analyze a list of sentences using the configured verifier.
        Runs analysis in parallel for better performance.
        
        Args:
            sentences: List of text sentences to analyze
            search_type: Type of search (basic/advanced)
            gpt_model: LLM model to use
            lang: UI language code
            progress_callback: Optional callback for progress updates
            analysis_mode: Analysis mode (general/lite)
            global_context: Pre-fetched context for the entire text
            global_sources: Sources associated with global context
            content_lang: M31 - Detected content language for query generation
            
        Returns:
            Dictionary with analysis results
        """
        if not self.verifier:
            raise ValueError("Verifier not configured for TextAnalyzer")
            
        
        total = len(sentences)
        
        async def process_sentence(idx, sent):
            # Pass progress_callback for first sentence (General Mode has 1 sentence)
            callback = progress_callback if idx == 0 else None
            
            # Context: previous sentence (if exists) to help with coreference resolution
            # e.g. "It is red." -> "The car is red."
            context_text = sentences[idx-1] if idx > 0 else ""
            
            # M31: Pass content_lang to verify_fact
            res = await self.verifier.verify_fact(
                sent, search_type, gpt_model, lang, analysis_mode, callback, 
                context_text=context_text,
                preloaded_context=global_context,
                preloaded_sources=global_sources,
                content_lang=content_lang  # M31
            )
            return idx, res

        tasks = [process_sentence(i, s) for i, s in enumerate(sentences)]
        
        results_with_index = []
        completed = 0
        
        if tasks:
            for future in asyncio.as_completed(tasks):
                idx, res = await future
                results_with_index.append((idx, res))
                completed += 1
                if progress_callback:
                    await progress_callback("analyzing", completed, total)
        
        # Restore original order
        results_with_index.sort(key=lambda x: x[0])
        results = [r for _, r in results_with_index]
        
        # Calculate total cost from individual results
        total_cost = sum(r.get("cost", 0) for r in results)
        
        # Aggregate sources from all results (for frontend display)
        all_sources = []
        seen_urls = set()
        for r in results:
            sources = r.get("sources", [])
            for source in sources:
                url = source.get("link", "")
                # Avoid duplicates by URL
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_sources.append(source)
        
        # Check if any result used search cache
        search_cache_hit = any(r.get("search_cache_hit", False) for r in results)
            
        return {
            "details": results, 
            "total_cost": total_cost,
            "sources": all_sources,
            "search_cache_hit": search_cache_hit
        }
