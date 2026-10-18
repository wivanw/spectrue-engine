# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""M147 T033: tier of unknown domains from the Decisions sensor. No network."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectrue_core.adapters.llm.domain_tier_sensor import CHANNELS, resolve_unknown_domains, tier_questions
from spectrue_core.domain.evidence.model import EvidenceChannel
from spectrue_core.llm.decisions import DecisionAnswer, DecisionResult
from spectrue_core.runtime_config import DecisionsConfig
from spectrue_core.tools import trusted_sources as ts


def test_questions_in_every_locale():
    for lang in ("en", "uk", "ru", "de", "es", "fr", "ja", "zh"):
        q = tier_questions(lang)[0].payload
        assert [c["value"] for c in q["choices"]] == list(CHANNELS) and "Prompt key" not in str(q)


@pytest.mark.asyncio
async def test_unknown_domain_gets_the_sensor_tier_known_domains_are_not_asked():
    llm = MagicMock()
    llm.call_decision = AsyncMock(return_value=DecisionResult(
        answers={"channel": DecisionAnswer(name="channel", type="choice", choice="reputable_news", confidence=0.9)},
        model="m", usage={}))
    domain = "some-unknown-news-site-147.ua"
    assert ts.get_domain_tier(domain) == EvidenceChannel.LOW_RELIABILITY
    with patch("spectrue_core.adapters.llm.decision_gates.Trace"):
        n = await resolve_unknown_domains(llm, DecisionsConfig(gate_modes={"domain_tier": "on"}),
                                          {domain: ["Drone attack on St Petersburg"], "reuters.com": ["x"]})
    assert n == 1 and llm.call_decision.call_count == 1          # reuters.com is on the curated list
    assert ts.get_domain_tier(domain) == EvidenceChannel.REPUTABLE_NEWS


@pytest.mark.asyncio
async def test_off_asks_nothing():
    llm = MagicMock(); llm.call_decision = AsyncMock()
    assert await resolve_unknown_domains(llm, DecisionsConfig(), {"x-unknown-147.org": ["t"]}) == 0
    assert not llm.call_decision.called
