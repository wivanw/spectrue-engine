# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Integration test for reranking search results on a cosmology query.

Uses a trace-derived query (DESI dark energy variation) and ensures
reranking keeps low-relevance results instead of discarding them.
"""

import pytest
from unittest.mock import AsyncMock

from spectrue_core.verification.search.search_mgr import SearchManager
from spectrue_core.verification.search.search_policy import SearchPolicyProfile, QualityThresholds, SearchProfileName


@pytest.mark.asyncio
async def test_search_unified_rerank_keeps_cosmology_results(mock_config):
    mock_config.google_fact_check_key = "test-google-fact-key"
    search_mgr = SearchManager(mock_config)
    profile = SearchPolicyProfile(
        name=SearchProfileName.DEEP.value,
        max_results=10,
        quality_thresholds=QualityThresholds(rerank_lambda=0.7),
    )
    search_mgr.set_policy_profile(profile)

    results = [
        {
            "title": "Dark Energy may be changing and with it the fate of the Universe - BBC",
            "link": "https://www.bbc.com/news/articles/c17xe5kl78vo",
            "snippet": "DESI data suggests dark energy could be evolving over time.",
            "score": 0.9,
            "relevance_score": 0.4,
        },
        {
            "title": "Astronomers discover one of the Universe’s largest spinning structures - ScienceDaily",
            "link": "https://www.sciencedaily.com/releases/2025/12/251225080729.htm",
            "snippet": "Large-scale structure studies complement dark energy surveys.",
            "score": 0.6,
            "relevance_score": 0.9,
        },
        {
            "title": "The top astronomical discoveries of 2025 - Space",
            "link": "https://www.space.com/astronomy/the-top-astronomical-discoveries-of-2025",
            "snippet": "A roundup that includes DESI updates.",
            "score": 0.4,
            "relevance_score": 0.8,
        },
        {
            "title": "Midwife-led training program improves outcomes",
            "link": "https://example.com/midwife-training",
            "snippet": "Unrelated healthcare training report.",
            "score": 0.35,
            "relevance_score": 0.05,
        },
    ]

    search_mgr.web_tool.search = AsyncMock(return_value=("", results))

    _, filtered = await search_mgr.search_unified(
        "DESI dark energy variation",
        topic="news",
        num_results=4,
    )

    assert len(filtered) == 4
    assert all("_rerank_score" in item for item in filtered)

    combined_scores = [
        0.7 * float(item.get("score")) + 0.3 * float(item.get("relevance_score"))
        for item in filtered
    ]
    assert combined_scores == sorted(combined_scores, reverse=True)
    assert any("Midwife-led" in item.get("title", "") for item in filtered)
