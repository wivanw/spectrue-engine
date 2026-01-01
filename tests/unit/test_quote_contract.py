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
Quote-Carrying Contract Unit Tests

Tests the invariant that quote fields are preserved through
the evidence matrix pipeline.
"""

from spectrue_core.agents.skills.clustering_parsing import (
    get_source_text_for_llm,
    build_sources_lite,
)


class TestGetSourceTextForLlm:
    """Test the canonical text extraction function."""

    def test_quote_takes_priority(self):
        """Quote field takes priority over snippet and content."""
        source = {
            "quote": "Direct quote from article",
            "snippet": "Snippet text",
            "content": "Full content",
        }
        text, has_quote, fields = get_source_text_for_llm(source)
        assert text == "Direct quote from article"
        assert has_quote is True
        assert "quote" in fields
        assert "snippet" in fields
        assert "content" in fields

    def test_fallback_to_snippet(self):
        """Falls back to snippet when no quote."""
        source = {"snippet": "Snippet text", "content": "Full content"}
        text, has_quote, fields = get_source_text_for_llm(source)
        assert text == "Snippet text"
        assert has_quote is False
        assert "snippet" in fields

    def test_fallback_to_content(self):
        """Falls back to content when no quote or snippet."""
        source = {"content": "Full content only"}
        text, has_quote, fields = get_source_text_for_llm(source)
        assert text == "Full content only"
        assert has_quote is False
        assert fields == ["content"]

    def test_extracted_content_fallback(self):
        """Falls back to extracted_content field."""
        source = {"extracted_content": "Extracted text"}
        text, has_quote, fields = get_source_text_for_llm(source)
        assert text == "Extracted text"
        assert has_quote is False
        assert fields == ["content"]

    def test_empty_fields_return_empty_string(self):
        """Empty source returns empty string."""
        source = {}
        text, has_quote, fields = get_source_text_for_llm(source)
        assert text == ""
        assert has_quote is False
        assert fields == []

    def test_max_len_truncation(self):
        """Text is truncated to max_len."""
        source = {"quote": "A" * 1000}
        text, has_quote, _ = get_source_text_for_llm(source, max_len=100)
        assert len(text) == 100
        assert has_quote is True

    def test_whitespace_stripping(self):
        """Whitespace is stripped from fields."""
        source = {"quote": "  quote with spaces  "}
        text, has_quote, _ = get_source_text_for_llm(source)
        assert text == "quote with spaces"
        assert has_quote is True


class TestBuildSourcesLiteQuoteContract:
    """Test the quote-carrying contract in build_sources_lite."""

    def test_quote_present_in_output_text(self):
        """Quote is present in output text field."""
        sources = [
            {
                "url": "https://example.com",
                "quote": "Important quote from source",
                "snippet": "Other text",
            }
        ]
        lite, _ = build_sources_lite(sources)
        assert lite[0]["has_quote"] is True
        assert "Important quote from source" in lite[0]["text"]
        assert lite[0]["fields_present"] == ["quote", "snippet", "quote_value"]

    def test_no_quote_uses_snippet(self):
        """Without quote, uses snippet."""
        sources = [{"url": "https://example.com", "snippet": "Snippet only"}]
        lite, _ = build_sources_lite(sources)
        assert lite[0]["has_quote"] is False
        assert "Snippet only" in lite[0]["text"]

    def test_multiple_sources_mixed_quotes(self):
        """Mixed sources: some with quotes, some without."""
        sources = [
            {"url": "https://a.com", "quote": "Quote A"},
            {"url": "https://b.com", "snippet": "Snippet B"},
            {"url": "https://c.com", "quote": "Quote C", "snippet": "Snippet C"},
        ]
        lite, _ = build_sources_lite(sources)

        assert lite[0]["has_quote"] is True
        assert "Quote A" in lite[0]["text"]

        assert lite[1]["has_quote"] is False
        assert "Snippet B" in lite[1]["text"]

        assert lite[2]["has_quote"] is True
        assert "Quote C" in lite[2]["text"]
        assert "Snippet C" not in lite[2]["text"]

    def test_unreadable_short_content_marked(self):
        """Short content is marked as unreadable."""
        sources = [{"url": "https://example.com", "snippet": "Hi"}]
        lite, unreadable = build_sources_lite(sources)
        assert 0 in unreadable
        assert "[UNREADABLE" in lite[0]["text"]

    def test_content_status_unavailable_hint(self):
        """Unavailable content gets status hint."""
        sources = [
            {
                "url": "https://example.com",
                "content_status": "unavailable",
                "snippet": "Snippet for unavailable page" + "x" * 50,
            }
        ]
        lite, _ = build_sources_lite(sources)
        assert "[CONTENT UNAVAILABLE" in lite[0]["text"]
