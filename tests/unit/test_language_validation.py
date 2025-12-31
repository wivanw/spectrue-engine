"""
Unit tests for language validation utilities (Phase 4).
"""

import pytest
from spectrue_core.utils.language_validation import (
    detect_claim_language,
    validate_claim_language,
    validate_claims_language_consistency,
)


class TestDetectClaimLanguage:
    """Test language detection."""
    
    def test_detect_ukrainian(self):
        """Ukrainian text should be detected as 'uk'."""
        text = "Українські війська звільнили місто на сході країни після важких боїв"
        lang, conf = detect_claim_language(text)
        assert lang == "uk"
        assert conf > 0.5
    
    def test_detect_english(self):
        """English text should be detected as 'en'."""
        text = "The president announced new economic policies during the press conference"
        lang, conf = detect_claim_language(text)
        assert lang == "en"
        assert conf > 0.5
    
    def test_short_text_returns_fallback(self):
        """Text < 20 chars should return fallback."""
        lang, conf = detect_claim_language("Short", fallback="uk")
        assert lang == "uk"
        assert conf == 0.0


class TestValidateClaimLanguage:
    """Test single claim language validation."""
    
    def test_matching_language_is_valid(self):
        """Claim in expected language should be valid."""
        claim = {"text": "Президент підписав новий закон про освіту"}
        is_valid, detected = validate_claim_language(claim, "uk")
        assert is_valid is True
    
    def test_mismatching_language_is_invalid(self):
        """Claim in wrong language should be invalid."""
        claim = {"text": "The president signed the new education law yesterday"}
        is_valid, detected = validate_claim_language(claim, "uk")
        # English text when expecting Ukrainian
        assert is_valid is False
        assert detected == "en"
    
    def test_short_text_is_always_valid(self):
        """Short text cannot be validated, so it's valid."""
        claim = {"text": "Short"}
        is_valid, detected = validate_claim_language(claim, "uk")
        assert is_valid is True


class TestValidateClaimsLanguageConsistency:
    """Test batch claim language validation."""
    
    def test_all_claims_match_is_valid(self):
        """All claims in correct language should pass."""
        claims = [
            {"id": "c1", "text": "Президент підписав закон про освіту"},
            {"id": "c2", "text": "Міністерство фінансів опублікувало звіт"},
        ]
        is_valid, mismatches = validate_claims_language_consistency(
            claims, "uk", pipeline_mode="deep"
        )
        assert is_valid is True
        assert len(mismatches) == 0
    
    def test_mismatch_in_normal_mode_fails(self):
        """Any mismatch in normal mode should fail."""
        claims = [
            {"id": "c1", "text": "The president signed the new law yesterday"},
        ]
        is_valid, mismatches = validate_claims_language_consistency(
            claims, "uk", pipeline_mode="normal"
        )
        assert is_valid is False
        assert len(mismatches) == 1
    
    def test_mismatch_in_deep_mode_allowed(self):
        """Mismatches in deep mode are logged but allowed."""
        claims = [
            {"id": "c1", "text": "Президент підписав закон"},
            {"id": "c2", "text": "The president signed the law"},  # English mixed in
        ]
        is_valid, mismatches = validate_claims_language_consistency(
            claims, "uk", pipeline_mode="deep"
        )
        # Deep mode allows mismatches (just logs them)
        assert is_valid is True
        assert len(mismatches) == 1
        assert mismatches[0]["claim_id"] == "c2"
    
    def test_empty_claims_is_valid(self):
        """Empty claims list should be valid."""
        is_valid, mismatches = validate_claims_language_consistency(
            [], "uk", pipeline_mode="normal"
        )
        assert is_valid is True
        assert len(mismatches) == 0

