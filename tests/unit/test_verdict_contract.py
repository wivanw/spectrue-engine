# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M71: Verdict Data Contract Tests - Pragmatic Edition.
"""

import json
import pytest
from pydantic import ValidationError

from spectrue_core.schema.policy import (
    ErrorState,
    DecisionPath,
    VerdictPolicy,
    DEFAULT_POLICY,
)

from spectrue_core.schema.signals import (
    RetrievalSignals,
    CoverageSignals,
    TimelinessSignals,
    EvidenceSignals,
)

from spectrue_core.schema.verdict_contract import (
    VerdictStatus,
    VerdictHighlight,
    Verdict,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Enum Serialization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnumSerialization:
    """Test JSON round-trip for all enums."""
    
    def test_error_state_json_roundtrip(self):
        for state in ErrorState:
            json_str = json.dumps(state.value)
            parsed = json.loads(json_str)
            restored = ErrorState(parsed)
            assert restored == state

    def test_decision_path_values(self):
        assert DecisionPath.ORACLE.value == "oracle"
        assert DecisionPath.WEB.value == "web"
        assert DecisionPath.CACHE.value == "cache"


# ═══════════════════════════════════════════════════════════════════════════════
# RetrievalSignals Invariant Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalSignalsInvariants:
    """Test RetrievalSignals invariants."""
    
    def test_considered_exceeds_found_fails(self):
        with pytest.raises(ValidationError, match="Considered > Found"):
            RetrievalSignals(
                total_sources_found=5,
                total_sources_considered=10,
            )
    
    def test_read_plus_unreadable_exceeds_considered_fails(self):
        with pytest.raises(ValidationError, match="Read \\+ Unreadable > Considered"):
            RetrievalSignals(
                total_sources_found=10,
                total_sources_considered=5,
                total_sources_read=3,
                unreadable_sources=3,
                unique_domains_count=3,  # 3+3=6 > 5
            )
    
    def test_sources_considered_but_zero_domains_fails(self):
        with pytest.raises(ValidationError, match="0 unique domains found"):
            RetrievalSignals(
                total_sources_found=10,
                total_sources_considered=5,
                total_sources_read=3,
                unique_domains_count=0,  # Must have at least 1
            )
    
    def test_valid_retrieval_passes(self):
        signals = RetrievalSignals(
            total_sources_found=15,
            total_sources_considered=10,
            total_sources_read=7,
            unreadable_sources=3,
            unique_domains_count=5,
        )
        assert signals.total_sources_read == 7
    
    def test_unreadable_breakdown_tracking(self):
        signals = RetrievalSignals(
            total_sources_found=10,
            total_sources_considered=8,
            total_sources_read=5,
            unreadable_sources=3,
            unreadable_breakdown={"paywall": 2, "timeout": 1},
            unique_domains_count=5,
        )
        assert signals.unreadable_breakdown["paywall"] == 2
        assert signals.unreadable_breakdown["timeout"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# CoverageSignals Invariant Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoverageSignalsInvariants:
    """Test CoverageSignals invariants."""
    
    def test_covered_exceeds_total_fails(self):
        with pytest.raises(ValidationError, match="Covered > Total"):
            CoverageSignals(
                assertions_total=5,
                assertions_covered=6,
            )
    
    def test_quotes_exceeds_covered_fails(self):
        with pytest.raises(ValidationError, match="Quotes > Covered"):
            CoverageSignals(
                assertions_total=10,
                assertions_covered=5,
                assertions_with_quotes=6,
            )
    
    def test_default_total_is_one(self):
        """Total defaults to 1 to avoid division by zero."""
        signals = CoverageSignals()
        assert signals.assertions_total == 1
    
    def test_valid_coverage_passes(self):
        signals = CoverageSignals(
            assertions_total=10,
            assertions_covered=8,
            assertions_with_quotes=5,
        )
        assert signals.assertions_with_quotes == 5


# ═══════════════════════════════════════════════════════════════════════════════
# TimelinessSignals Invariant Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimelinessSignalsInvariants:
    """Test TimelinessSignals chronology invariant."""
    
    def test_newest_older_than_oldest_fails(self):
        with pytest.raises(ValidationError, match="Newest age > Oldest age"):
            TimelinessSignals(
                newest_source_age_hours=48.0,  # Newest is 2 days old
                oldest_source_age_hours=24.0,  # But oldest is only 1 day old?
            )
    
    def test_valid_chronology_passes(self):
        signals = TimelinessSignals(
            newest_source_age_hours=2.0,
            oldest_source_age_hours=48.0,
        )
        assert signals.newest_source_age_hours == 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# EvidenceSignals Property Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvidenceSignalsProperties:
    """Test EvidenceSignals derived properties."""
    
    def test_has_readable_sources_true(self):
        signals = EvidenceSignals(
            retrieval=RetrievalSignals(
                total_sources_found=10,
                total_sources_considered=8,
                total_sources_read=5,
                unique_domains_count=5,
            )
        )
        assert signals.has_readable_sources is True
    
    def test_has_readable_sources_false(self):
        signals = EvidenceSignals()
        assert signals.has_readable_sources is False
    
    def test_coverage_ratio(self):
        signals = EvidenceSignals(
            coverage=CoverageSignals(
                assertions_total=10,
                assertions_covered=6,
            )
        )
        assert signals.coverage_ratio == 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict Score Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerdictScoreValidation:
    """Test Verdict score bounds."""
    
    def test_veracity_above_one_fails(self):
        with pytest.raises(ValidationError):
            Verdict(veracity_score=1.5)
    
    def test_confidence_below_zero_fails(self):
        with pytest.raises(ValidationError):
            Verdict(confidence_score=-0.1)
    
    def test_default_confidence_is_zero(self):
        """M71 Critical: Confidence = 0.0 by default (Blind until proven)."""
        verdict = Verdict()
        assert verdict.confidence_score == 0.0
    
    def test_default_veracity_is_half(self):
        verdict = Verdict()
        assert verdict.veracity_score == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Status Derivation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatusDerivation:
    """Test Verdict.status() derivation logic."""
    
    def test_error_state_returns_unknown(self):
        for error in [
            ErrorState.NO_EVIDENCE_RETRIEVED,
            ErrorState.EVIDENCE_UNREADABLE,
            ErrorState.UPSTREAM_TIMEOUT,
            ErrorState.PIPELINE_ERROR,
        ]:
            verdict = Verdict(
                veracity_score=0.9,
                confidence_score=0.9,
                error_state=error,
            )
            assert verdict.status() == VerdictStatus.UNKNOWN
    
    def test_high_confidence_verified(self):
        verdict = Verdict(
            veracity_score=0.9,
            confidence_score=0.85,
            signals=EvidenceSignals(
                coverage=CoverageSignals(
                    assertions_total=5,
                    assertions_covered=5,
                    assertions_with_quotes=3,  # Has quotes!
                )
            ),
        )
        assert verdict.status() == VerdictStatus.VERIFIED
    
    def test_high_confidence_refuted(self):
        verdict = Verdict(
            veracity_score=0.1,
            confidence_score=0.85,
            signals=EvidenceSignals(
                coverage=CoverageSignals(
                    assertions_total=5,
                    assertions_covered=5,
                    assertions_with_quotes=3,
                )
            ),
        )
        assert verdict.status() == VerdictStatus.REFUTED
    
    def test_low_confidence_returns_ambiguous(self):
        verdict = Verdict(
            veracity_score=0.9,
            confidence_score=0.4,  # Below 0.8 threshold
        )
        assert verdict.status() == VerdictStatus.AMBIGUOUS
    
    def test_middle_veracity_returns_ambiguous(self):
        verdict = Verdict(
            veracity_score=0.5,
            confidence_score=0.9,
        )
        assert verdict.status() == VerdictStatus.AMBIGUOUS
    
    def test_no_quotes_high_confidence_forced_ambiguous(self):
        """Safety: If no quotes, high confidence is forced AMBIGUOUS."""
        verdict = Verdict(
            veracity_score=0.9,
            confidence_score=0.9,  # High confidence
            signals=EvidenceSignals(
                coverage=CoverageSignals(
                    assertions_total=5,
                    assertions_covered=5,
                    assertions_with_quotes=0,  # No quotes!
                )
            ),
        )
        # Should be forced to AMBIGUOUS due to no quotes
        assert verdict.status() == VerdictStatus.AMBIGUOUS
    
    def test_default_verdict_is_ambiguous(self):
        """Default Verdict (confidence=0.0) should be AMBIGUOUS."""
        verdict = Verdict()
        assert verdict.status() == VerdictStatus.AMBIGUOUS


# ═══════════════════════════════════════════════════════════════════════════════
# Policy Customization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolicyCustomization:
    """Test custom policy thresholds."""
    
    def test_lenient_policy(self):
        lenient = VerdictPolicy(
            min_confidence_for_verified=0.6,
            verified_veracity_threshold=0.7,
            max_confidence_without_quotes=None,  # Disable quote requirement
        )
        
        verdict = Verdict(
            veracity_score=0.75,
            confidence_score=0.65,
        )
        
        # Default policy: AMBIGUOUS (below 0.8/0.8)
        assert verdict.status(DEFAULT_POLICY) == VerdictStatus.AMBIGUOUS
        
        # Lenient policy: VERIFIED
        assert verdict.status(lenient) == VerdictStatus.VERIFIED
    
    def test_disable_quote_safety(self):
        """Custom policy can disable quote safety check."""
        no_quote_check = VerdictPolicy(
            max_confidence_without_quotes=None,  # Disable
        )
        
        verdict = Verdict(
            veracity_score=0.9,
            confidence_score=0.9,
            signals=EvidenceSignals(
                coverage=CoverageSignals(
                    assertions_total=5,
                    assertions_covered=5,
                    assertions_with_quotes=0,  # No quotes
                )
            ),
        )
        
        # Default policy: AMBIGUOUS (no quotes)
        assert verdict.status(DEFAULT_POLICY) == VerdictStatus.AMBIGUOUS
        
        # Custom policy: VERIFIED (quote check disabled)
        assert verdict.status(no_quote_check) == VerdictStatus.VERIFIED


# ═══════════════════════════════════════════════════════════════════════════════
# Model Serialization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelSerialization:
    """Test JSON round-trip for models."""
    
    def test_verdict_json_roundtrip(self):
        original = Verdict(
            veracity_score=0.85,
            confidence_score=0.75,
            error_state=ErrorState.OK,
            decision_path=DecisionPath.WEB,
            summary="Test verdict",
        )
        
        json_str = original.model_dump_json()
        restored = Verdict.model_validate_json(json_str)
        
        assert restored.veracity_score == original.veracity_score
        assert restored.confidence_score == original.confidence_score
    
    def test_status_not_serialized(self):
        verdict = Verdict()
        dumped = verdict.model_dump()
        
        assert "status" not in dumped
        assert "status_default" not in dumped
