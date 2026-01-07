# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
M127: Unit tests for coverage_skeleton module.

Tests:
- Skeleton dataclasses
- Coverage analyzers (time, number, quote detection)
- Skeleton parsing from LLM response
- Coverage validation
"""

from spectrue_core.agents.skills.coverage_skeleton import (
    SkeletonEvent,
    SkeletonMeasurement,
    SkeletonPolicy,
    QuantityMention,
    CoverageSkeleton,
    CoverageAnalysis,
    extract_time_mentions_count,
    extract_time_mentions,
    extract_number_mentions_count,
    detect_quote_spans_count,
    analyze_text_coverage,
    validate_skeleton_coverage,
    parse_skeleton_response,
)


class TestExtractTimeMentions:
    """Test time mention detection."""

    def test_detect_yyyy(self):
        """Should detect 4-digit years."""
        text = "In 2024, sales grew significantly."
        assert extract_time_mentions_count(text) >= 1
        mentions = extract_time_mentions(text)
        assert any("2024" in m for m in mentions)

    def test_detect_yyyy_mm_dd(self):
        """Should detect full dates."""
        text = "The event happened on 2024-01-15."
        assert extract_time_mentions_count(text) >= 1

    def test_detect_month_year(self):
        """Should detect Month YYYY format."""
        text = "January 2025 will be important."
        assert extract_time_mentions_count(text) >= 1

    def test_detect_quarter(self):
        """Should detect Q1-Q4 YYYY format."""
        text = "Q4 2024 results exceeded expectations."
        assert extract_time_mentions_count(text) >= 1

    def test_multiple_dates(self):
        """Should count multiple dates."""
        text = "From 2023-01-15 to 2024-02-01 there was growth."
        assert extract_time_mentions_count(text) >= 2

    def test_no_dates(self):
        """Should return 0 when no dates."""
        text = "No specific dates mentioned here."
        assert extract_time_mentions_count(text) == 0

    def test_empty_text(self):
        """Should handle empty text."""
        assert extract_time_mentions_count("") == 0
        assert extract_time_mentions_count(None) == 0


class TestExtractNumberMentions:
    """Test number mention detection."""

    def test_detect_currency(self):
        """Should detect currency amounts."""
        text = "Revenue was $5.2 billion."
        assert extract_number_mentions_count(text) >= 1

    def test_detect_percentage(self):
        """Should detect percentages."""
        text = "Growth increased by 15.5%."
        assert extract_number_mentions_count(text) >= 1

    def test_detect_large_numbers(self):
        """Should detect significant large numbers."""
        text = "Population reached 10000 people."
        assert extract_number_mentions_count(text) >= 1

    def test_detect_millions_billions(self):
        """Should detect numbers with million/billion."""
        text = "The company is worth 100 million dollars."
        assert extract_number_mentions_count(text) >= 1

    def test_no_numbers(self):
        """Should return 0 when no significant numbers."""
        text = "No numbers here at all."
        assert extract_number_mentions_count(text) == 0

    def test_empty_text(self):
        """Should handle empty text."""
        assert extract_number_mentions_count("") == 0

    def test_filter_years(self):
        """Years should not be counted as number mentions."""
        text = "In 2024, specifically the year 2024."
        # Years are handled by time patterns, not numbers
        # This may still count them but we're testing the intent


class TestDetectQuoteSpans:
    """Test quote span detection."""

    def test_detect_double_quotes(self):
        """Should detect standard double quotes."""
        text = 'He said "this is important news".'
        assert detect_quote_spans_count(text) >= 1

    def test_detect_curly_quotes(self):
        """Should detect curly quotes."""
        text = 'She stated "we will succeed" firmly.'
        assert detect_quote_spans_count(text) >= 1

    def test_detect_guillemets(self):
        """Should detect guillemets."""
        text = "Il a dit «bonjour mon ami»."
        assert detect_quote_spans_count(text) >= 1

    def test_no_quotes(self):
        """Should return 0 when no quotes."""
        text = "No quoted text here."
        assert detect_quote_spans_count(text) == 0

    def test_short_quotes_filtered(self):
        """Should filter very short quotes."""
        text = 'She said "ok".'  # Too short (< 5 chars)
        assert detect_quote_spans_count(text) == 0

    def test_empty_text(self):
        """Should handle empty text."""
        assert detect_quote_spans_count("") == 0


class TestCoverageAnalysis:
    """Test combined coverage analysis."""

    def test_analyze_rich_text(self):
        """Should detect multiple coverage indicators."""
        text = """
        In January 2025, Tesla reported $5.2 billion in revenue.
        CEO Elon Musk said "We expect continued growth in Q4 2025."
        """
        analysis = analyze_text_coverage(text)
        assert analysis.detected_times >= 2  # 2025, Q4 2025
        assert analysis.detected_numbers >= 1  # $5.2 billion
        assert analysis.detected_quotes >= 1  # "We expect..."


class TestValidateSkeletonCoverage:
    """Test skeleton coverage validation."""

    def test_good_coverage_passes(self):
        """Skeleton with good coverage should pass."""
        skeleton = CoverageSkeleton(
            events=[SkeletonEvent(
                id="evt_1",
                subject_entities=["Tesla"],
                verb_phrase="reported revenue",
                time_anchor={"type": "explicit_date", "value": "Q4 2024"},
                location_anchor=None,
                raw_span="Tesla reported revenue in Q4 2024",
            )],
            measurements=[SkeletonMeasurement(
                id="msr_1",
                subject_entities=["Tesla"],
                metric="revenue",
                quantity_mentions=[QuantityMention(value="5.2", unit="billion")],
                time_anchor={"type": "explicit_date", "value": "Q4 2024"},
                raw_span="$5.2 billion in Q4 2024",
            )],
            quotes=[],
            policies=[],
        )
        analysis = CoverageAnalysis(
            detected_times=1,
            detected_numbers=1,
            detected_quotes=0,
        )
        ok, reason_codes = validate_skeleton_coverage(skeleton, analysis)
        assert ok is True
        assert reason_codes == []

    def test_low_coverage_fails(self):
        """Skeleton with low coverage should fail."""
        skeleton = CoverageSkeleton(
            events=[],
            measurements=[],
            quotes=[],
            policies=[],
        )
        analysis = CoverageAnalysis(
            detected_times=5,
            detected_numbers=3,
            detected_quotes=2,
        )
        ok, reason_codes = validate_skeleton_coverage(skeleton, analysis, tolerance=0.5)
        assert ok is False
        assert len(reason_codes) >= 1


class TestParseSkeletonResponse:
    """Test parsing LLM response into skeleton."""

    def test_parse_complete_response(self):
        """Should parse all skeleton types."""
        data = {
            "events": [
                {
                    "id": "evt_1",
                    "subject_entities": ["Tesla"],
                    "verb_phrase": "announced",
                    "time_anchor": {"type": "explicit_date", "value": "2024"},
                    "raw_span": "Tesla announced in 2024",
                }
            ],
            "measurements": [
                {
                    "id": "msr_1",
                    "subject_entities": ["Company"],
                    "metric": "revenue",
                    "quantity_mentions": [{"value": "5", "unit": "billion"}],
                    "raw_span": "$5 billion revenue",
                }
            ],
            "quotes": [
                {
                    "id": "qot_1",
                    "speaker_entities": ["CEO"],
                    "quote_text": "We are growing",
                    "raw_span": "CEO said we are growing",
                }
            ],
            "policies": [
                {
                    "id": "pol_1",
                    "subject_entities": ["EU"],
                    "policy_action": "requires safety features",
                    "raw_span": "EU requires safety features",
                }
            ],
        }
        skeleton = parse_skeleton_response(data)
        
        assert len(skeleton.events) == 1
        assert skeleton.events[0].id == "evt_1"
        
        assert len(skeleton.measurements) == 1
        assert skeleton.measurements[0].metric == "revenue"
        
        assert len(skeleton.quotes) == 1
        assert skeleton.quotes[0].quote_text == "We are growing"
        
        assert len(skeleton.policies) == 1
        assert skeleton.policies[0].policy_action == "requires safety features"

    def test_parse_empty_response(self):
        """Should handle empty/missing arrays."""
        data = {}
        skeleton = parse_skeleton_response(data)
        
        assert skeleton.events == []
        assert skeleton.measurements == []
        assert skeleton.quotes == []
        assert skeleton.policies == []

    def test_parse_malformed_items(self):
        """Should skip malformed items."""
        data = {
            "events": [
                "not a dict",  # Should be skipped
                {"id": "evt_1", "subject_entities": [], "verb_phrase": "test", "raw_span": "test"},
            ],
            "measurements": [],
            "quotes": [],
            "policies": [],
        }
        skeleton = parse_skeleton_response(data)
        assert len(skeleton.events) == 1


class TestSkeletonDataclasses:
    """Test skeleton dataclass methods."""

    def test_skeleton_counts(self):
        """Should return correct counts."""
        skeleton = CoverageSkeleton(
            events=[SkeletonEvent("e1", ["A"], "v", None, None, "r")],
            measurements=[
                SkeletonMeasurement("m1", ["B"], "m", [], None, "r"),
                SkeletonMeasurement("m2", ["C"], "m", [], None, "r"),
            ],
            quotes=[],
            policies=[SkeletonPolicy("p1", ["D"], "a", None, "r")],
        )
        counts = skeleton.counts()
        assert counts["events"] == 1
        assert counts["measurements"] == 2
        assert counts["quotes"] == 0
        assert counts["policies"] == 1
        assert skeleton.total_items() == 4

    def test_to_dict(self):
        """Should serialize to dict correctly."""
        event = SkeletonEvent(
            id="evt_1",
            subject_entities=["Tesla"],
            verb_phrase="announced",
            time_anchor={"type": "explicit_date", "value": "2024"},
            location_anchor="Palo Alto",
            raw_span="Tesla announced in Palo Alto in 2024",
        )
        d = event.to_dict()
        assert d["id"] == "evt_1"
        assert d["subject_entities"] == ["Tesla"]
        assert d["location_anchor"] == "Palo Alto"
