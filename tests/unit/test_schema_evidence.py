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
Schema Evidence Tests

Tests for EvidenceItem, ContentStatus handling, and per-assertion evidence mapping.
"""

from spectrue_core.schema import (
    EvidenceItem,
    EvidenceStance,
    ContentStatus,
)


class TestEvidenceItemSchema:
    """Tests for EvidenceItem structure."""

    def test_basic_evidence_item(self):
        """Test creating basic evidence item."""
        evidence = EvidenceItem(
            claim_id="c1",
            assertion_key="event.location.city",
            stance=EvidenceStance.SUPPORT,
            excerpt="The fight will take place in Miami",
            domain="espn.com",
            url="https://espn.com/article",
        )
        
        assert evidence.claim_id == "c1"
        assert evidence.assertion_key == "event.location.city"
        assert evidence.stance == EvidenceStance.SUPPORT
        assert evidence.is_actionable() is True

    def test_sentinel_default_scores(self):
        """Test that scores default to -1.0 sentinel."""
        evidence = EvidenceItem(claim_id="c1")
        
        assert evidence.retrieval_confidence == -1.0, "retrieval_confidence should be -1.0 sentinel"
        assert evidence.relevance_score == -1.0, "relevance_score should be -1.0 sentinel"
        
        # Sentinel detection
        assert evidence.retrieval_confidence < 0, "Sentinel should be detectable as < 0"
        assert evidence.relevance_score < 0, "Sentinel should be detectable as < 0"

    def test_actual_scores_override_sentinel(self):
        """Test that actual scores work correctly."""
        evidence = EvidenceItem(
            claim_id="c1",
            retrieval_confidence=0.95,
            relevance_score=0.85,
        )
        
        assert evidence.retrieval_confidence == 0.95
        assert evidence.relevance_score == 0.85
        assert evidence.retrieval_confidence >= 0


class TestContentUnavailableHandling:
    """
    T14: Bug Reproduction Test - OFAC Empty Snippet.
    
    When content can't be retrieved (treasury.gov/OFAC), evidence should:
    - Be kept with CONTENT_UNAVAILABLE status
    - Lower explainability_score
    - Claim stays AMBIGUOUS, NOT REFUTED
    """

    def test_content_unavailable_status(self):
        """Test CONTENT_UNAVAILABLE evidence creation."""
        evidence = EvidenceItem(
            claim_id="c1",
            assertion_key="sanctions.status",
            stance=EvidenceStance.MENTION,
            domain="treasury.gov",
            url="https://treasury.gov/ofac/list",
            content_status=ContentStatus.CONTENT_UNAVAILABLE,
            unavailable_reason="Auth wall - content behind login",
        )
        
        assert evidence.content_status == ContentStatus.CONTENT_UNAVAILABLE
        assert evidence.unavailable_reason is not None
        
        # CONTENT_UNAVAILABLE is NOT actionable for verdict
        assert evidence.is_actionable() is False
        
        # But it DOES affect explainability
        assert evidence.affects_explainability() is True

    def test_blocked_status(self):
        """Test BLOCKED evidence (429, 403 errors)."""
        evidence = EvidenceItem(
            claim_id="c1",
            content_status=ContentStatus.BLOCKED,
            unavailable_reason="HTTP 429 Too Many Requests",
        )
        
        assert evidence.content_status == ContentStatus.BLOCKED
        assert evidence.is_actionable() is False
        assert evidence.affects_explainability() is True

    def test_available_content_is_actionable(self):
        """Test that AVAILABLE content with valid stance is actionable."""
        evidence = EvidenceItem(
            claim_id="c1",
            stance=EvidenceStance.SUPPORT,
            content_status=ContentStatus.AVAILABLE,
        )
        
        assert evidence.is_actionable() is True
        assert evidence.affects_explainability() is False

    def test_mention_stance_not_actionable(self):
        """MENTION stance is not actionable even with AVAILABLE content."""
        evidence = EvidenceItem(
            claim_id="c1",
            stance=EvidenceStance.MENTION,
            content_status=ContentStatus.AVAILABLE,
        )
        
        assert evidence.is_actionable() is False

    def test_irrelevant_stance_not_actionable(self):
        """IRRELEVANT stance is not actionable."""
        evidence = EvidenceItem(
            claim_id="c1",
            stance=EvidenceStance.IRRELEVANT,
            content_status=ContentStatus.AVAILABLE,
        )
        
        assert evidence.is_actionable() is False


class TestEvidenceStance:
    """Tests for EvidenceStance enum."""

    def test_valid_stances(self):
        """Test all valid stance values."""
        assert EvidenceStance.SUPPORT.value == "SUPPORT"
        assert EvidenceStance.REFUTE.value == "REFUTE"
        assert EvidenceStance.MIXED.value == "MIXED"
        assert EvidenceStance.MENTION.value == "MENTION"
        assert EvidenceStance.IRRELEVANT.value == "IRRELEVANT"

    def test_actionable_stances(self):
        """Only SUPPORT, REFUTE, MIXED are actionable for verdicts."""
        actionable = [EvidenceStance.SUPPORT, EvidenceStance.REFUTE, EvidenceStance.MIXED]
        not_actionable = [EvidenceStance.MENTION, EvidenceStance.IRRELEVANT]
        
        for stance in actionable:
            ev = EvidenceItem(claim_id="c1", stance=stance, content_status=ContentStatus.AVAILABLE)
            assert ev.is_actionable() is True, f"{stance} should be actionable"
        
        for stance in not_actionable:
            ev = EvidenceItem(claim_id="c1", stance=stance, content_status=ContentStatus.AVAILABLE)
            assert ev.is_actionable() is False, f"{stance} should NOT be actionable"


class TestAssertionKeyMapping:
    """Tests for assertion_key evidence mapping."""

    def test_assertion_key_for_location(self):
        """Evidence can map to specific location assertion."""
        evidence = EvidenceItem(
            claim_id="c1",
            assertion_key="event.location.city",
            stance=EvidenceStance.SUPPORT,
            quote="confirmed for Miami Gardens venue",
        )
        
        assert evidence.assertion_key == "event.location.city"

    def test_assertion_key_for_time(self):
        """Evidence can map to specific time assertion."""
        evidence = EvidenceItem(
            claim_id="c1",
            assertion_key="event.time",
            stance=EvidenceStance.SUPPORT,
            quote="fight starts at 03:00",
        )
        
        assert evidence.assertion_key == "event.time"

    def test_empty_assertion_key_means_whole_claim(self):
        """Empty assertion_key means evidence applies to whole claim (legacy mode)."""
        evidence = EvidenceItem(
            claim_id="c1",
            assertion_key="",  # Empty = whole claim
        )
        
        assert evidence.assertion_key == ""

    def test_multiple_evidence_for_same_assertion(self):
        """Multiple evidence items can map to same assertion_key."""
        evidence1 = EvidenceItem(
            claim_id="c1",
            assertion_key="event.location.city",
            stance=EvidenceStance.SUPPORT,
            domain="espn.com",
        )
        evidence2 = EvidenceItem(
            claim_id="c1",
            assertion_key="event.location.city",
            stance=EvidenceStance.SUPPORT,
            domain="bbc.com",
        )
        
        # Both map to same assertion
        assert evidence1.assertion_key == evidence2.assertion_key
        assert evidence1.domain != evidence2.domain
