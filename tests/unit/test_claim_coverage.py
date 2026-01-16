# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Tests for Deterministic Coverage Anchors and Hard Guards.

Tests cover:
- Time anchor extraction (dates in various formats)
- Numeric anchor extraction (numbers with context)
- Quote anchor extraction (quoted spans)
- Instructions guard (fallback injection)
- Coverage validation
"""

from spectrue_core.verification.claims.coverage_anchors import (
    AnchorKind,
    extract_time_anchors,
    extract_numeric_anchors,
    extract_quote_anchors,
    extract_all_anchors,
    get_anchor_ids,
    anchors_to_prompt_context,
)
from spectrue_core.agents.skills.claims import (
    DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS,
)


class TestTimeAnchors:
    """Test time anchor extraction."""

    def test_extract_iso_date(self):
        """Extract ISO format dates: YYYY-MM-DD."""
        text = "The event occurred on 2024-01-15 and continued until 2024-01-20."
        anchors = extract_time_anchors(text)
        
        assert len(anchors) == 2
        assert anchors[0].kind == AnchorKind.TIME
        assert "2024-01-15" in anchors[0].span_text
        assert "2024-01-20" in anchors[1].span_text

    def test_extract_year_only(self):
        """Extract standalone years."""
        text = "This happened in 2023. By 2025, everything changed."
        anchors = extract_time_anchors(text)
        
        assert len(anchors) == 2
        assert anchors[0].span_text == "2023"
        assert anchors[1].span_text == "2025"

    def test_extract_quarter_format(self):
        """Extract quarter format: Q1 2024."""
        text = "Revenue grew in Q4 2024 compared to Q1 2023."
        anchors = extract_time_anchors(text)
        
        assert len(anchors) >= 2
        spans = [a.span_text for a in anchors]
        assert any("Q4" in s and "2024" in s for s in spans)
        assert any("Q1" in s and "2023" in s for s in spans)

    def test_extract_month_year(self):
        """Extract month year format: January 2024."""
        text = "The announcement came in January 2025 and February 2025."
        anchors = extract_time_anchors(text)
        
        assert len(anchors) >= 2
        spans = [a.span_text for a in anchors]
        assert any("January" in s and "2025" in s for s in spans)

    def test_context_window_captured(self):
        """Verify context window is captured around anchors."""
        text = "The important meeting was scheduled for 2024-03-15 at noon."
        anchors = extract_time_anchors(text)
        
        assert len(anchors) >= 1
        assert "meeting" in anchors[0].context_window or "scheduled" in anchors[0].context_window


class TestNumericAnchors:
    """Test numeric anchor extraction."""

    def test_extract_currency(self):
        """Extract currency values."""
        text = "The company raised $5 million and spent €2.3 billion."
        anchors = extract_numeric_anchors(text)
        
        assert len(anchors) >= 2
        spans = [a.span_text for a in anchors]
        assert any("$" in s for s in spans)

    def test_extract_percentage(self):
        """Extract percentage values."""
        text = "Sales increased by 15% while costs rose only 3.5%."
        anchors = extract_numeric_anchors(text)
        
        assert len(anchors) >= 2
        spans = [a.span_text for a in anchors]
        assert any("15%" in s for s in spans)
        assert any("3.5%" in s for s in spans)

    def test_extract_large_numbers(self):
        """Extract large formatted numbers."""
        text = "The population reached 1,234,567 people."
        anchors = extract_numeric_anchors(text)
        
        assert len(anchors) >= 1
        assert "1,234,567" in anchors[0].span_text

    def test_extract_numbers_with_units(self):
        """Extract numbers with units."""
        text = "The distance is 100km and temperature was 25°C."
        anchors = extract_numeric_anchors(text)
        
        assert len(anchors) >= 2


class TestQuoteAnchors:
    """Test quote anchor extraction."""

    def test_extract_double_quotes(self):
        """Extract text in double quotes."""
        text = 'The CEO said "We expect growth" during the meeting.'
        anchors = extract_quote_anchors(text)
        
        assert len(anchors) >= 1
        assert "We expect growth" in anchors[0].span_text

    def test_extract_curly_quotes(self):
        """Extract text in curly quotes."""
        text = 'She stated "This is important" at the conference.'
        anchors = extract_quote_anchors(text)
        
        assert len(anchors) >= 1
        assert "This is important" in anchors[0].span_text

    def test_skip_short_quotes(self):
        """Skip very short quotes (likely punctuation artifacts)."""
        text = 'He said "ok" but then added "This is the real statement we need".'
        anchors = extract_quote_anchors(text)
        
        # Should skip "ok" (too short) but capture the longer quote
        assert all(len(a.span_text) >= 5 for a in anchors)


class TestCombinedExtraction:
    """Test combined anchor extraction."""

    def test_extract_all_anchor_types(self):
        """Extract all anchor types from realistic text."""
        text = '''
        On 2024-01-15, Tesla announced revenue of $5.2 billion.
        CEO Elon Musk said "We expect 2025 to be transformational"
        during the earnings call.
        '''
        anchors = extract_all_anchors(text)
        
        # Should have time anchors, numeric anchors, and quote anchors
        time_anchors = [a for a in anchors if a.kind == AnchorKind.TIME]
        numeric_anchors = [a for a in anchors if a.kind == AnchorKind.NUMBER]
        quote_anchors = [a for a in anchors if a.kind == AnchorKind.QUOTE]
        
        assert len(time_anchors) >= 1, "Should extract at least one date"
        assert len(numeric_anchors) >= 1, "Should extract at least one number"
        assert len(quote_anchors) >= 1, "Should extract at least one quote"

    def test_anchor_ids_are_unique(self):
        """Verify anchor IDs are unique across types."""
        text = "In 2024, revenue was $5 billion. The CEO said \"Growth continues\"."
        anchors = extract_all_anchors(text)
        
        ids = get_anchor_ids(anchors)
        assert len(ids) == len(anchors), "All anchor IDs should be unique"

    def test_anchors_sorted_by_position(self):
        """Verify anchors are sorted by char_start position."""
        text = "First 2024, then $100, finally \"a quote\"."
        anchors = extract_all_anchors(text)
        
        positions = [a.char_start for a in anchors]
        assert positions == sorted(positions), "Anchors should be sorted by position"


class TestAnchorHelpers:
    """Test helper functions."""

    def test_get_anchor_ids(self):
        """Test get_anchor_ids returns set of IDs."""
        text = "In 2024, the price was $50."
        anchors = extract_all_anchors(text)
        
        ids = get_anchor_ids(anchors)
        assert isinstance(ids, set)
        assert all(isinstance(id, str) for id in ids)

    def test_anchors_to_prompt_context(self):
        """Test prompt context formatting."""
        text = "In 2024, the price was $50."
        anchors = extract_all_anchors(text)
        
        context = anchors_to_prompt_context(anchors)
        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain anchor IDs
        for anchor in anchors:
            assert anchor.anchor_id in context


class TestInstructionsGuard:
    """Test hard guard for instructions."""

    def test_default_instructions_defined(self):
        """Verify DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS is defined and non-empty."""
        assert DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS is not None
        assert len(DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS) > 100
        
    def test_default_instructions_contains_key_rules(self):
        """Verify instructions contain key extraction rules."""
        instructions = DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS
        
        assert "extracting" in instructions.lower()
        assert "verifiable" in instructions.lower() or "factual" in instructions.lower()
        assert "json schema" in instructions.lower()

    def test_default_instructions_prefer_over_extraction(self):
        """Verify instructions prefer over-extraction."""
        instructions = DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS
        
        assert "over-extraction" in instructions.lower()


class TestNoMaxClaimsCap:
    """Test that extraction has no artificial caps."""

    def test_no_max_claims_in_default_instructions(self):
        """Verify default instructions don't mention claim limits."""
        instructions = DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS
        
        # Should NOT contain limiting language
        assert "top" not in instructions.lower() or "top-k" not in instructions.lower()
        assert "maximum" not in instructions.lower() or "maximum 5" not in instructions.lower()
        assert "at most" not in instructions.lower()


class TestCoverageValidation:
    """Test coverage validation functions."""

    def test_validate_coverage_all_covered(self):
        """No gap when all anchors are covered."""
        from spectrue_core.verification.claims.coverage_validator import (
            validate_coverage,
        )
        
        text = "Revenue was $50 billion."
        anchors = extract_all_anchors(text)
        anchor_ids = [a.anchor_id for a in anchors]
        
        # Create skeleton items that cover ALL anchors dynamically
        skeleton_items = [{"anchor_refs": anchor_ids}]
        
        gap = validate_coverage(anchors, skeleton_items, [])
        assert gap is None

    def test_validate_coverage_with_skipped(self):
        """Skipped anchors are not gaps."""
        from spectrue_core.verification.claims.coverage_validator import (
            validate_coverage,
        )
        
        text = "Revenue was $50 billion."
        anchors = extract_all_anchors(text)
        
        # Cover first anchor, skip the rest
        skeleton_items = [{"anchor_refs": [anchors[0].anchor_id]}] if anchors else []
        skipped = [{"anchor_id": a.anchor_id, "reason_code": "not_a_fact"} for a in anchors[1:]]
        
        gap = validate_coverage(anchors, skeleton_items, skipped)
        assert gap is None

    def test_validate_coverage_detects_gaps(self):
        """Detect missing anchors."""
        from spectrue_core.verification.claims.coverage_validator import (
            validate_coverage,
        )
        
        text = "Revenue was $50 billion."
        anchors = extract_all_anchors(text)
        
        # Don't cover any anchors - should detect gaps
        skeleton_items = []
        
        gap = validate_coverage(anchors, skeleton_items, [])
        assert gap is not None
        assert len(gap.missing_anchor_ids) == len(anchors)


class TestGapFillHelpers:
    """Test gap-fill helper functions."""

    def test_build_gapfill_prompt(self):
        """Verify gap-fill prompt structure."""
        from spectrue_core.verification.claims.coverage_validator import (
            build_gapfill_prompt,
        )
        
        text = "In 2024, revenue was $50 billion."
        anchors = extract_all_anchors(text)
        missing = [a for a in anchors if a.anchor_id == "n1"]
        
        skeleton = {"events": [], "measurements": [], "quotes": [], "policies": []}
        
        prompt = build_gapfill_prompt(text, skeleton, missing)
        
        assert "GAP-FILL" in prompt
        assert "n1" in prompt

    def test_merge_gapfill_result(self):
        """Verify gapfill merge works correctly."""
        from spectrue_core.verification.claims.coverage_validator import (
            merge_gapfill_result,
        )
        
        original = {
            "events": [{"id": "evt_1"}],
            "measurements": [],
            "quotes": [],
            "policies": [],
            "skipped_anchors": [],
        }
        
        gapfill = {
            "new_measurements": [{"id": "msr_1", "anchor_refs": ["n1"]}],
            "additional_skipped_anchors": [],
        }
        
        merged = merge_gapfill_result(original, gapfill)
        
        assert len(merged["events"]) == 1
        assert len(merged["measurements"]) == 1
        assert merged["measurements"][0]["id"] == "msr_1"

    def test_check_remaining_gaps(self):
        """Verify remaining gap detection after merge."""
        from spectrue_core.verification.claims.coverage_validator import (
            check_remaining_gaps,
        )
        
        text = "Revenue was $50 billion."
        anchors = extract_all_anchors(text)
        anchor_ids = [a.anchor_id for a in anchors]
        
        # Complete skeleton (all covered dynamically)
        complete_skeleton = {
            "events": [{"anchor_refs": anchor_ids}],
            "measurements": [],
            "quotes": [],
            "policies": [],
            "skipped_anchors": [],
        }
        
        remaining = check_remaining_gaps(anchors, complete_skeleton)
        assert remaining == []
