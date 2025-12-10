# Simplified engine.py without TextAnalyzer dependency

import logging
from typing import Dict, Any
from langdetect import detect, DetectorFactory, LangDetectException
import re

from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.fact_verifier_composite import FactVerifierComposite

# Make language detection deterministic
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

def detect_content_language(text: str, fallback: str = "en") -> str:
    """Detect language of input text using langdetect library."""
    clean_text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    clean_text = clean_text.strip()
    
    if len(clean_text) < 20:
        return fallback
    
    try:
        detected_lang = detect(clean_text)
        
        supported = ["en", "uk", "ru", "de", "es", "fr", "ja", "zh"]
        if detected_lang in supported:
            return detected_lang
        
        lang_mapping = {
            "pt": "es", "it": "es", "ca": "es",
            "nl": "de", "pl": "uk", "cs": "uk", "sk": "uk", "be": "uk",
            "bg": "ru", "sr": "ru",
            "ko": "ja", "vi": "zh", "th": "zh",
        }
        
        return lang_mapping.get(detected_lang, fallback)
        
    except LangDetectException:
        return fallback


class SpectrueEngine:
    """The main entry point for the Spectrue Fact-Checking Engine."""
    
    def __init__(self, config: SpectrueConfig):
        self.config = config
        self.verifier = FactVerifierComposite(config)

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
        Analyze text with content detection and waterfall verification.
        
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
        
        # Detect content language
        content_lang = detect_content_language(text, fallback=lang)
        
        # Call verifier directly
        result = await self.verifier.verify_fact(
            fact=text,
            search_type=search_type,
            gpt_model=model,
            lang=lang,
            analysis_mode=analysis_mode,
            progress_callback=progress_callback,
            content_lang=content_lang
        )
        
        return result
