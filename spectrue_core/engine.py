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

import logging
from typing import Dict, Any, Optional
from langdetect import detect, DetectorFactory, LangDetectException
import re

from spectrue_core.config import SpectrueConfig
from spectrue_core.analysis.text_analyzer import TextAnalyzer
from spectrue_core.verification.fact_verifier_composite import FactVerifierComposite

# Make language detection deterministic
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

def detect_content_language(text: str, fallback: str = "en") -> str:
    """
    Detect language of input text using langdetect library.
    """
    clean_text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    clean_text = clean_text.strip()
    
    # Need at least 20 chars for reliable detection
    if len(clean_text) < 20:
        return fallback
    
    try:
        detected_lang = detect(clean_text)
        
        # Map to supported languages (8 locales)
        supported = ["en", "uk", "ru", "de", "es", "fr", "ja", "zh"]
        if detected_lang in supported:
            return detected_lang
        
        # Fallback mapping for dialects/similar languages
        lang_mapping = {
            "pt": "es",  # Portuguese -> Spanish (similar sources)
            "it": "es",  # Italian -> Spanish
            "ca": "es",  # Catalan -> Spanish
            "nl": "de",  # Dutch -> German
            "pl": "uk",  # Polish -> Ukrainian (Eastern Europe)
            "cs": "uk",  # Czech -> Ukrainian
            "sk": "uk",  # Slovak -> Ukrainian
            "be": "uk",  # Belarusian -> Ukrainian
            "bg": "ru",  # Bulgarian -> Russian (Cyrillic)
            "sr": "ru",  # Serbian -> Russian (Cyrillic)
            "ko": "ja",  # Korean -> Japanese (East Asia)
            "vi": "zh",  # Vietnamese -> Chinese (Southeast Asia)
            "th": "zh",  # Thai -> Chinese (Southeast Asia)
        }
        
        return lang_mapping.get(detected_lang, fallback)
        
    except LangDetectException:
        return fallback


class SpectrueEngine:
    """
    The main entry point for the Spectrue Fact-Checking Engine.
    """
    
    def __init__(self, config: SpectrueConfig):
        self.config = config
        
        # Initialize components with config
        self.verifier = FactVerifierComposite(config)
        
        # Determine max sentences config
        # TODO: Move max_sentences to SpectrueConfig if needed
        analyzer_config = {
            'max_sentences': 24, # Default
        }
        
        self.text_analyzer = TextAnalyzer(self.verifier, analyzer_config)

    async def analyze_text(
        self, 
        text: str,
        lang: str = "en",
        analysis_mode: str = "general",
        gpt_model: str = None,
        search_type: str = "advanced",
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Analyze logic with content detection and waterfall verification.
        
        Args:
            text: Text to analyze
            lang: UI language ISO code
            analysis_mode: "general" or "lite"
            gpt_model: Model override (optional, uses config default if None)
            search_type: "basic" or "advanced"
            progress_callback: Async callable(stage: str)
            
        Returns:
            Dict with analysis result
        """
        model = gpt_model or self.config.openai_model
        
        # M31: Detect content language
        content_lang = detect_content_language(text, fallback=lang)
        
        # Call TextAnalyzer
        result = await self.text_analyzer.process(
            text=text,
            lang=lang,
            content_lang=content_lang,
            analysis_mode=analysis_mode,
            gpt_model=model,
            search_type=search_type,
            progress_callback=progress_callback
        )
        
        return result
