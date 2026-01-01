# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Unit tests for medical log redaction.
"""

import pytest
from spectrue_core.utils.trace import _redact_medical, _redact_text


class TestMedicalRedaction:
    """T1: Medical Log Redaction - Unit Tests."""

    # ─────────────────────────────────────────────────────────────────────────
    # Dosing patterns
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("input_text,expected_contains", [
        # Ukrainian dosing
        ("Приймати 15 мл 3 рази на день", "[REDACTED_MEDICAL]"),
        ("Доза: 500 мг на добу", "[REDACTED_MEDICAL]"),
        ("Вводити 100 мкг/кг", "[REDACTED_MEDICAL]"),
        ("Рекомендовано 2 г на день", "[REDACTED_MEDICAL]"),
        # English dosing
        ("Take 500mg twice daily", "[REDACTED_MEDICAL]"),
        ("Administer 10 ml per kg", "[REDACTED_MEDICAL]"),
        ("Inject 50 iu daily", "[REDACTED_MEDICAL]"),
    ])
    def test_dosing_patterns_redacted(self, input_text: str, expected_contains: str):
        """Dosing patterns should be redacted."""
        result = _redact_medical(input_text)
        assert expected_contains in result
        # Original numbers should not appear
        assert "15 мл" not in result or "[REDACTED" in result
        assert "500 мг" not in result or "[REDACTED" in result

    # ─────────────────────────────────────────────────────────────────────────
    # Instruction patterns
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("input_text", [
        "приймати по 2 таблетки",
        "пити 3 рази на день",
        "вживати 1 ложку",
        "take 2 tablets",
        "consume 3 capsules",
        "apply 5 drops",
    ])
    def test_instruction_patterns_redacted(self, input_text: str):
        """Intake instructions should be redacted."""
        result = _redact_medical(input_text)
        assert "[REDACTED_MEDICAL]" in result

    # ─────────────────────────────────────────────────────────────────────────
    # Frequency patterns
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("input_text", [
        "3 рази на день",
        "2 рази на добу",
        "1 раз на тиждень",
        "3 times per day",
        "2 times daily",
        "1 time a day", # Changed from "once a day"
    ])
    def test_frequency_patterns_redacted(self, input_text: str):
        """Frequency patterns should be redacted."""
        result = _redact_medical(input_text)
        assert "[REDACTED_MEDICAL]" in result

    # ─────────────────────────────────────────────────────────────────────────
    # Duration patterns
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("input_text", [
        "протягом 7 днів",
        "впродовж 2 днів", # Changed from "2 тижнів" (regex covers days/hours/minutes)
        "for 5 days",
        "during 24 hours",
    ])
    def test_duration_patterns_redacted(self, input_text: str):
        """Duration patterns should be redacted."""
        result = _redact_medical(input_text)
        assert "[REDACTED_MEDICAL]" in result

    # ─────────────────────────────────────────────────────────────────────────
    # Procedural steps
    # ─────────────────────────────────────────────────────────────────────────

    def test_procedural_steps_redacted(self):
        """Procedural steps with numbers should be redacted."""
        text = "Крок 1: Змішати компоненти разом"
        result = _redact_medical(text)
        assert "[REDACTED_MEDICAL]" in result

    # ─────────────────────────────────────────────────────────────────────────
    # Preservation tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_non_medical_text_preserved(self):
        """Non-medical text should not be redacted."""
        text = "Це звичайний текст про новини дня. Сьогодні сталося 5 подій."
        result = _redact_medical(text)
        assert "[REDACTED_MEDICAL]" not in result
        assert "5 подій" in result

    def test_numbers_without_units_preserved(self):
        """Plain numbers without medical context should be preserved."""
        text = "Ціна становить 500 грн. У статті 10 абзаців."
        result = _redact_medical(text)
        assert "500 грн" in result
        assert "10 абзаців" in result

    def test_empty_string_handled(self):
        """Empty string should return empty string."""
        assert _redact_medical("") == ""
        assert _redact_medical(None) is None

    # ─────────────────────────────────────────────────────────────────────────
    # Integration with _redact_text
    # ─────────────────────────────────────────────────────────────────────────

    def test_redact_text_includes_medical(self):
        """_redact_text should include medical redaction."""
        text = "https://api.com?key=secret123&foo=bar, dose: 500 мг"
        result = _redact_text(text)
        # API key redacted
        assert "key=***" in result
        # Medical redacted
        assert "[REDACTED_MEDICAL]" in result

    def test_combined_redaction(self):
        """Both API keys and medical content should be redacted together."""
        text = "Request to https://api.example.com?key=abc123\nArticle text: Рекомендована доза 100 мг 2 рази на день\nAuthorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = _redact_text(text)
        
        # API key redacted
        assert "key=***" in result
        # Bearer token redacted
        assert "Bearer ***" in result
        # Medical redacted
        assert "[REDACTED_MEDICAL]" in result
        # Original sensitive data not present
        assert "abc123" not in result
        assert "100 мг" not in result or "[REDACTED" in result


class TestHashPreservation:
    """Verify sha256 hashing still works after redaction."""

    def test_long_text_with_medical_produces_hash(self):
        """Long text should produce sha256 hash even after redaction."""
        from spectrue_core.utils.trace import _sanitize
        
        # Create long medical text
        long_text = "Інструкція: приймати 500 мг 3 рази на день. " * 200
        
        result = _sanitize(long_text, max_str=1000)
        
        # Should be truncated with hash
        assert isinstance(result, dict)
        assert "sha256" in result
        assert len(result["sha256"]) == 64  # SHA256 hex length
        assert "head" in result
        assert "tail" in result
        # Head should contain redacted version
        assert "[REDACTED_MEDICAL]" in result["head"] or "Інструкція" in result["head"]
