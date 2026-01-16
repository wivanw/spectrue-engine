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

"""Tests for PhaseRunner batch enrichment."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def make_mock_search_mgr():
    """Create a mock SearchManager with fetch_urls_content_batch."""
    search_mgr = MagicMock()
    search_mgr.fetch_urls_content_batch = AsyncMock(return_value={
        "https://a.com": "Content A with enough text to be useful.",
        "https://b.com": "Content B with enough text to be useful.",
        "https://c.com": "Content C with enough text to be useful.",
        "https://d.com": "Content D with enough text to be useful.",
    })
    return search_mgr


class TestPhaseRunnerBatchEnrichment:
    """Tests for PhaseRunner._batch_enrich_all_sources."""

    @pytest.mark.asyncio
    async def test_batch_enriches_all_claim_sources_at_once(self):
        """All URLs from all claims should be fetched in one batch call."""
        from spectrue_core.verification.orchestration.phase_runner import PhaseRunner

        search_mgr = make_mock_search_mgr()
        runner = PhaseRunner(search_mgr)

        evidence = {
            "c1": [
                {"url": "https://a.com", "snippet": "short"},
                {"url": "https://b.com", "snippet": "short"},
            ],
            "c2": [
                {"url": "https://c.com", "snippet": "short"},
            ],
        }

        result = await runner._batch_enrich_all_sources(evidence)

        # Should call fetch_urls_content_batch exactly once
        assert search_mgr.fetch_urls_content_batch.call_count == 1

        # Should pass all 3 unique URLs
        call_args = search_mgr.fetch_urls_content_batch.call_args
        urls_passed = call_args[0][0]  # First positional argument
        assert len(urls_passed) == 3
        assert set(urls_passed) == {"https://a.com", "https://b.com", "https://c.com"}

        # Sources should be enriched
        assert result["c1"][0]["fulltext"] is True
        assert result["c1"][1]["fulltext"] is True
        assert result["c2"][0]["fulltext"] is True

    @pytest.mark.asyncio
    async def test_batch_includes_inline_sources(self):
        """Inline sources should be included in the batch fetch."""
        from spectrue_core.verification.orchestration.phase_runner import PhaseRunner

        search_mgr = make_mock_search_mgr()
        inline_sources = [
            {"url": "https://d.com", "snippet": "inline source"},
        ]
        runner = PhaseRunner(search_mgr, inline_sources=inline_sources)

        evidence = {
            "c1": [
                {"url": "https://a.com", "snippet": "short"},
            ],
        }

        await runner._batch_enrich_all_sources(evidence)

        # Should call with both claim URL and inline URL
        call_args = search_mgr.fetch_urls_content_batch.call_args
        urls_passed = call_args[0][0]
        assert len(urls_passed) == 2
        assert set(urls_passed) == {"https://a.com", "https://d.com"}

        # Inline source should also be enriched
        assert inline_sources[0].get("fulltext") is True

    @pytest.mark.asyncio
    async def test_deduplicates_urls_across_claims(self):
        """Same URL in multiple claims should only be fetched once."""
        from spectrue_core.verification.orchestration.phase_runner import PhaseRunner

        search_mgr = make_mock_search_mgr()
        runner = PhaseRunner(search_mgr)

        evidence = {
            "c1": [{"url": "https://a.com", "snippet": "short"}],
            "c2": [{"url": "https://a.com", "snippet": "short"}],  # Same URL
            "c3": [{"url": "https://b.com", "snippet": "short"}],
        }

        await runner._batch_enrich_all_sources(evidence)

        # Should only fetch 2 unique URLs
        call_args = search_mgr.fetch_urls_content_batch.call_args
        urls_passed = call_args[0][0]
        assert len(urls_passed) == 2
        assert set(urls_passed) == {"https://a.com", "https://b.com"}

        # Both sources with same URL should be enriched
        assert evidence["c1"][0].get("fulltext") is True
        assert evidence["c2"][0].get("fulltext") is True

    @pytest.mark.asyncio
    async def test_skips_already_enriched_sources(self):
        """Sources with fulltext=True should not be re-enriched."""
        from spectrue_core.verification.orchestration.phase_runner import PhaseRunner

        search_mgr = make_mock_search_mgr()
        runner = PhaseRunner(search_mgr)

        original_content = "Original content that should not change"
        evidence = {
            "c1": [
                {"url": "https://a.com", "content": original_content, "fulltext": True},
            ],
        }

        await runner._batch_enrich_all_sources(evidence)

        # Content should not be overwritten
        assert evidence["c1"][0]["content"] == original_content

    @pytest.mark.asyncio
    async def test_handles_empty_evidence(self):
        """Empty evidence dict should not cause errors."""
        from spectrue_core.verification.orchestration.phase_runner import PhaseRunner

        search_mgr = make_mock_search_mgr()
        runner = PhaseRunner(search_mgr)

        result = await runner._batch_enrich_all_sources({})

        # Should not call fetch
        assert search_mgr.fetch_urls_content_batch.call_count == 0
        assert result == {}
