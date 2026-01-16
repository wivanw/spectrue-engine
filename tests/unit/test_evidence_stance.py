# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for evidence_stance.py module.

Tests the stance classification and conflict detection extracted in M119.
"""

import pytest
from unittest.mock import patch, MagicMock

from spectrue_core.verification.evidence.evidence_stance import (
    CANONICAL_VERDICT_STATES,
    count_stance_evidence,
    derive_verdict_state_from_llm_score,
    derive_verdict_from_score,
    detect_evidence_conflict,
    check_has_direct_evidence,
    assign_claim_rgba,
)
from spectrue_core.pipeline.mode import ScoringMode


# ─────────────────────────────────────────────────────────────────────────────
# Constants Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConstants:
    """Tests for module constants."""

    def test_canonical_verdict_states(self):
        """Should have expected verdict states."""
        assert "supported" in CANONICAL_VERDICT_STATES
        assert "refuted" in CANONICAL_VERDICT_STATES
        assert "conflicted" in CANONICAL_VERDICT_STATES
        assert "insufficient_evidence" in CANONICAL_VERDICT_STATES
        assert len(CANONICAL_VERDICT_STATES) == 4


# ─────────────────────────────────────────────────────────────────────────────
# count_stance_evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCountStanceEvidence:
    """Tests for count_stance_evidence function."""

    def test_counts_support_evidence(self):
        """Should count supporting evidence."""
        evidence = [
            {"claim_id": "c1", "stance": "support"},
            {"claim_id": "c1", "stance": "support"},
            {"claim_id": "c1", "stance": "neutral"},
        ]
        n_support, n_refute, _ = count_stance_evidence("c1", evidence)
        assert n_support == 2
        assert n_refute == 0

    def test_counts_refute_evidence(self):
        """Should count refuting evidence."""
        evidence = [
            {"claim_id": "c1", "stance": "refute"},
            {"claim_id": "c1", "stance": "ref"},  # Abbreviated form
        ]
        n_support, n_refute, _ = count_stance_evidence("c1", evidence)
        assert n_support == 0
        assert n_refute == 2

    def test_counts_mixed_evidence(self):
        """Should count both support and refute."""
        evidence = [
            {"claim_id": "c1", "stance": "support"},
            {"claim_id": "c1", "stance": "refute"},
            {"claim_id": "c1", "stance": "support"},
        ]
        n_support, n_refute, _ = count_stance_evidence("c1", evidence)
        assert n_support == 2
        assert n_refute == 1

    def test_filters_by_claim_id(self):
        """Should only count evidence for specified claim."""
        evidence = [
            {"claim_id": "c1", "stance": "support"},
            {"claim_id": "c2", "stance": "support"},
            {"claim_id": "c1", "stance": "refute"},
        ]
        n_support, n_refute, _ = count_stance_evidence("c1", evidence)
        assert n_support == 1
        assert n_refute == 1

    def test_includes_evidence_without_claim_id(self):
        """Evidence without claim_id should be included."""
        evidence = [
            {"claim_id": None, "stance": "support"},
            {"stance": "support"},  # No claim_id key
        ]
        n_support, n_refute, _ = count_stance_evidence("c1", evidence)
        assert n_support == 2

    def test_tracks_best_tier(self):
        """Should return best tier from evidence."""
        evidence = [
            {"claim_id": "c1", "stance": "support", "tier": "C"},
            {"claim_id": "c1", "stance": "support", "tier": "A"},
            {"claim_id": "c1", "stance": "support", "tier": "B"},
        ]
        _, _, best_tier = count_stance_evidence("c1", evidence)
        assert best_tier == "A"

    def test_handles_empty_evidence(self):
        """Should handle empty evidence list."""
        n_support, n_refute, best_tier = count_stance_evidence("c1", [])
        assert n_support == 0
        assert n_refute == 0
        assert best_tier is None

    def test_handles_non_dict_items(self):
        """Should skip non-dict items."""
        evidence = [
            {"claim_id": "c1", "stance": "support"},
            None,
            "invalid",
            123,
        ]
        n_support, n_refute, _ = count_stance_evidence("c1", evidence)
        assert n_support == 1

    def test_case_insensitive_stance(self):
        """Should handle stance case-insensitively."""
        evidence = [
            {"claim_id": "c1", "stance": "SUPPORT"},
            {"claim_id": "c1", "stance": "Support"},
            {"claim_id": "c1", "stance": "REFUTE"},
        ]
        n_support, n_refute, _ = count_stance_evidence("c1", evidence)
        assert n_support == 2
        assert n_refute == 1


# ─────────────────────────────────────────────────────────────────────────────
# derive_verdict_state_from_llm_score Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeriveVerdictStateFromLlmScore:
    """Tests for derive_verdict_state_from_llm_score function."""

    def test_high_score_supported(self):
        """Score > 0.65 should return 'supported'."""
        assert derive_verdict_state_from_llm_score(0.8, 0, 0) == "supported"
        assert derive_verdict_state_from_llm_score(0.66, 0, 0) == "supported"

    def test_low_score_refuted(self):
        """Score < 0.35 should return 'refuted'."""
        assert derive_verdict_state_from_llm_score(0.2, 0, 0) == "refuted"
        assert derive_verdict_state_from_llm_score(0.34, 0, 0) == "refuted"

    def test_middle_score_with_evidence_conflicted(self):
        """Middle score with evidence should return 'conflicted'."""
        assert derive_verdict_state_from_llm_score(0.5, 1, 0) == "conflicted"
        assert derive_verdict_state_from_llm_score(0.5, 0, 1) == "conflicted"
        assert derive_verdict_state_from_llm_score(0.5, 2, 3) == "conflicted"

    def test_middle_score_without_evidence_insufficient(self):
        """Middle score without evidence should return 'insufficient_evidence'."""
        assert derive_verdict_state_from_llm_score(0.5, 0, 0) == "insufficient_evidence"
        assert derive_verdict_state_from_llm_score(0.4, 0, 0) == "insufficient_evidence"

    def test_boundary_values(self):
        """Test boundary values."""
        # Exactly at threshold
        assert derive_verdict_state_from_llm_score(0.65, 0, 0) == "insufficient_evidence"
        assert derive_verdict_state_from_llm_score(0.35, 0, 0) == "insufficient_evidence"


# ─────────────────────────────────────────────────────────────────────────────
# derive_verdict_from_score Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDeriveVerdictFromScore:
    """Tests for derive_verdict_from_score function."""

    def test_high_score_verified(self):
        """Score > 0.65 should return 'verified'."""
        assert derive_verdict_from_score(0.9) == "verified"
        assert derive_verdict_from_score(0.66) == "verified"

    def test_low_score_refuted(self):
        """Score < 0.35 should return 'refuted'."""
        assert derive_verdict_from_score(0.1) == "refuted"
        assert derive_verdict_from_score(0.34) == "refuted"

    def test_middle_score_ambiguous(self):
        """Score between 0.35 and 0.65 should return 'ambiguous'."""
        assert derive_verdict_from_score(0.5) == "ambiguous"
        assert derive_verdict_from_score(0.4) == "ambiguous"
        assert derive_verdict_from_score(0.6) == "ambiguous"

    def test_boundary_values(self):
        """Test boundary values."""
        assert derive_verdict_from_score(0.35) == "ambiguous"
        assert derive_verdict_from_score(0.65) == "ambiguous"


# ─────────────────────────────────────────────────────────────────────────────
# detect_evidence_conflict Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectEvidenceConflict:
    """Tests for detect_evidence_conflict function."""

    def test_conflict_when_both_present(self):
        """Should detect conflict when both support and refute exist."""
        assert detect_evidence_conflict(1, 1) is True
        assert detect_evidence_conflict(5, 2) is True

    def test_no_conflict_support_only(self):
        """Should not detect conflict with only support."""
        assert detect_evidence_conflict(3, 0) is False

    def test_no_conflict_refute_only(self):
        """Should not detect conflict with only refute."""
        assert detect_evidence_conflict(0, 2) is False

    def test_no_conflict_no_evidence(self):
        """Should not detect conflict with no evidence."""
        assert detect_evidence_conflict(0, 0) is False


# ─────────────────────────────────────────────────────────────────────────────
# check_has_direct_evidence Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckHasDirectEvidence:
    """Tests for check_has_direct_evidence function."""

    def test_has_direct_support_with_quote(self):
        """Should detect direct evidence with SUPPORT stance and quote."""
        evidence = [
            {"claim_id": "c1", "stance": "SUPPORT", "quote": "This is a quote."},
        ]
        assert check_has_direct_evidence("c1", evidence) is True

    def test_has_direct_refute_with_quote(self):
        """Should detect direct evidence with REFUTE stance and quote."""
        evidence = [
            {"claim_id": "c1", "stance": "REFUTE", "quote": "Contradicting quote."},
        ]
        assert check_has_direct_evidence("c1", evidence) is True

    def test_no_direct_without_quote(self):
        """Should not detect direct evidence without quote."""
        evidence = [
            {"claim_id": "c1", "stance": "SUPPORT", "quote": None},
            {"claim_id": "c1", "stance": "SUPPORT"},  # No quote key
        ]
        assert check_has_direct_evidence("c1", evidence) is False

    def test_no_direct_neutral_stance(self):
        """Should not detect direct evidence with NEUTRAL stance."""
        evidence = [
            {"claim_id": "c1", "stance": "NEUTRAL", "quote": "Some quote."},
        ]
        assert check_has_direct_evidence("c1", evidence) is False

    def test_filters_by_claim_id(self):
        """Should only check evidence for specified claim."""
        evidence = [
            {"claim_id": "c1", "stance": "SUPPORT", "quote": "Quote for c1."},
            {"claim_id": "c2", "stance": "SUPPORT", "quote": "Quote for c2."},
        ]
        assert check_has_direct_evidence("c1", evidence) is True
        # c3 has no evidence
        assert check_has_direct_evidence("c3", evidence) is False

    def test_handles_empty_evidence(self):
        """Should return False for empty evidence."""
        assert check_has_direct_evidence("c1", []) is False


# ─────────────────────────────────────────────────────────────────────────────
# assign_claim_rgba Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAssignClaimRgba:
    """Tests for assign_claim_rgba function."""

    @pytest.fixture
    def mock_trace(self):
        """Mock Trace to avoid side effects."""
        with patch("spectrue_core.utils.trace.Trace") as mock:
            mock.event = MagicMock()
            yield mock

    def test_preserves_existing_valid_rgba(self, mock_trace):
        """Should preserve existing valid RGBA."""
        claim_verdict = {
            "claim_id": "c1",
            "verdict_score": 0.8,
            "rgba": [0.1, 0.9, 0.7, 0.6],
        }
        assign_claim_rgba(
            claim_verdict,
            global_r=0.2,
            global_b=0.5,
            global_a=0.8,
            judge_mode=ScoringMode.DEEP,
        )
        # Should keep original RGBA
        assert claim_verdict["rgba"] == [0.1, 0.9, 0.7, 0.6]

    def test_standard_mode_computes_rgba(self, mock_trace):
        """Standard mode should compute RGBA from globals."""
        claim_verdict = {
            "claim_id": "c1",
            "verdict_score": 0.8,
        }
        assign_claim_rgba(
            claim_verdict,
            global_r=0.2,
            global_b=0.5,
            global_a=0.8,
            judge_mode=ScoringMode.STANDARD,
        )
        # Should have computed RGBA
        assert "rgba" in claim_verdict

    def test_deep_mode_missing_rgba_marked_error(self, mock_trace):
        """Deep mode claim without RGBA should be marked as error."""
        claim_verdict = {
            "claim_id": "c1",
            "verdict_score": 0.8,
            "rgba": None,
        }
        assign_claim_rgba(
            claim_verdict,
            global_r=0.2,
            global_b=0.5,
            global_a=0.8,
            judge_mode=ScoringMode.DEEP,
        )
        # Should mark as error
        assert claim_verdict.get("rgba_error") == "missing_from_llm"

    def test_deep_mode_error_claim_no_rgba_expected(self, mock_trace):
        """Deep mode error claim should not require RGBA."""
        claim_verdict = {
            "claim_id": "c1",
            "status": "error",
            "error_type": "llm_failed",
        }
        assign_claim_rgba(
            claim_verdict,
            global_r=0.2,
            global_b=0.5,
            global_a=0.8,
            judge_mode=ScoringMode.DEEP,
        )
        # Should not add rgba_error for error claims
        assert "rgba_error" not in claim_verdict

    def test_invalid_rgba_replaced(self, mock_trace):
        """Invalid RGBA format should be replaced."""
        claim_verdict = {
            "claim_id": "c1",
            "verdict_score": 0.8,
            "rgba": [0.1, 0.9],  # Only 2 elements - invalid
        }
        assign_claim_rgba(
            claim_verdict,
            global_r=0.2,
            global_b=0.5,
            global_a=0.8,
            judge_mode=ScoringMode.STANDARD,
        )
        # Should have computed new RGBA
        rgba = claim_verdict.get("rgba")
        assert rgba is None or len(rgba) == 4

