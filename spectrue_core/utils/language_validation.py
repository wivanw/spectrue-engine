"""
Language validation utilities for claim processing.

Implements invariants for language consistency:
1. Search queries use original claim.text language
2. normalized_text is ONLY for display/embedding
3. Language mismatch in normal pipeline = fail-fast
"""

import logging
from typing import Tuple

from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


def detect_claim_language(text: str, fallback: str = "en") -> Tuple[str, float]:
    """
    Detect language of claim text.
    
    Returns:
        Tuple of (language_code, confidence)
    """
    from langdetect import detect_langs, LangDetectException
    import re

    # Clean URLs and mentions
    clean_text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    clean_text = clean_text.strip()

    if len(clean_text) < 20:
        return fallback, 0.0

    try:
        langs = detect_langs(clean_text)
        if not langs:
            return fallback, 0.0
        detected_lang = langs[0].lang
        confidence = float(getattr(langs[0], "prob", 0.0) or 0.0)

        supported = ["en", "uk", "ru", "de", "es", "fr", "ja", "zh"]
        if detected_lang in supported:
            return detected_lang, confidence

        return fallback, confidence

    except LangDetectException:
        return fallback, 0.0


def validate_claim_language(
    claim: dict,
    expected_lang: str,
    *,
    min_confidence: float = 0.7,
    claim_id: str | None = None,
) -> Tuple[bool, str | None]:
    """
    Validate that claim text is in expected language.
    
    Args:
        claim: Claim dict with 'text' field
        expected_lang: Expected language code (e.g., 'uk', 'en')
        min_confidence: Minimum confidence for detection
        claim_id: For tracing
        
    Returns:
        Tuple of (is_valid, detected_lang)
        - is_valid: True if language matches or detection uncertain
        - detected_lang: Detected language code or None
    """
    text = claim.get("text") or ""
    if not text or len(text) < 20:
        return True, None  # Too short to validate

    detected_lang, confidence = detect_claim_language(text)

    # If detection is uncertain, allow it
    if confidence < min_confidence:
        return True, detected_lang

    # Check for mismatch
    is_match = detected_lang == expected_lang

    if not is_match:
        Trace.event(
            "pipeline.language_mismatch",
            {
                "claim_id": claim_id,
                "expected_lang": expected_lang,
                "detected_lang": detected_lang,
                "confidence": round(confidence, 3),
                "text_preview": text[:100],
            },
        )
        logger.warning(
            "Language mismatch for claim %s: expected=%s, detected=%s (conf=%.2f)",
            claim_id,
            expected_lang,
            detected_lang,
            confidence,
        )

    return is_match, detected_lang


def validate_claims_language_consistency(
    claims: list[dict],
    expected_lang: str,
    *,
    pipeline_mode: str = "normal",
    min_confidence: float = 0.7,
) -> Tuple[bool, list[dict]]:
    """
    Validate language consistency across all claims.
    
    In normal mode (single-claim), any mismatch is a violation.
    In deep mode (multi-claim), log mismatches but allow processing.
    
    Args:
        claims: List of claim dicts
        expected_lang: Expected language code
        pipeline_mode: 'normal' or 'deep'
        min_confidence: Minimum confidence for detection
        
    Returns:
        Tuple of (all_valid, mismatches)
        - all_valid: True if no critical mismatches
        - mismatches: List of mismatch details
    """
    mismatches = []

    for claim in claims:
        claim_id = claim.get("id") or claim.get("claim_id")
        is_valid, detected_lang = validate_claim_language(
            claim,
            expected_lang,
            min_confidence=min_confidence,
            claim_id=claim_id,
        )

        if not is_valid:
            mismatches.append({
                "claim_id": claim_id,
                "expected_lang": expected_lang,
                "detected_lang": detected_lang,
            })

    # In normal mode, any mismatch is critical
    if pipeline_mode == "normal" and mismatches:
        Trace.event(
            "pipeline.language_violation",
            {
                "mode": pipeline_mode,
                "expected_lang": expected_lang,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
            },
        )
        return False, mismatches

    return True, mismatches

