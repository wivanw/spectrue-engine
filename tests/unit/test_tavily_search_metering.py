# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tavily search metering is depth-aware (advanced = 2 credits)."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from spectrue_core.billing.config_loader import load_pricing_policy
from spectrue_core.billing.cost_ledger import CostLedger
from spectrue_core.billing.metering import TavilyMeter
from spectrue_core.tools.tavily_client import TavilyClient, tavily_search_credits


def _response(body: dict):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = body
    return response


@pytest.mark.parametrize(
    "response,depth,expected",
    [
        ({"results": []}, "advanced", 2.0),
        ({"results": []}, "ADVANCED", 2.0),
        ({"results": []}, "basic", 1.0),
        ({"results": []}, "fast", 1.0),
        ({"results": []}, None, 1.0),
        ({"results": [], "usage": {"credits": 2}}, "basic", 2.0),
        ({"results": [], "usage": {"credits": 1}}, "advanced", 1.0),
        ({"results": [], "usage": {"credits": "bad"}}, "advanced", 2.0),
        ({"results": [], "usage": {}}, "advanced", 2.0),
        (None, "advanced", 2.0),
    ],
)
def test_tavily_search_credits(response, depth, expected):
    assert tavily_search_credits(response, depth) == expected


async def _run_search(depth: str, body: dict):
    ledger = CostLedger(run_id="test")
    policy = load_pricing_policy()
    client = TavilyClient(api_key="test-key", meter=TavilyMeter(ledger=ledger, policy=policy))
    mock_post = AsyncMock(return_value=_response(body))
    client._client.post = mock_post
    await client.search(query="q", depth=depth, max_results=5)
    await client.close()
    return ledger, mock_post, policy


@pytest.mark.asyncio
async def test_advanced_search_bills_two_credits_without_usage():
    ledger, mock_post, policy = await _run_search("advanced", {"results": []})

    payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
    assert payload["include_usage"] is True

    assert len(ledger.events) == 1
    event = ledger.events[0]
    assert event.meta["credits_used"] == 2.0
    assert event.meta["search_depth"] == "advanced"
    basic = TavilyMeter(ledger=CostLedger(run_id="x"), policy=policy).record_search(credits_used=1.0)
    assert Decimal(event.cost_credits) == 2 * Decimal(basic.cost_credits)


@pytest.mark.asyncio
async def test_basic_search_bills_one_credit():
    ledger, _, _ = await _run_search("basic", {"results": []})
    assert ledger.events[0].meta["credits_used"] == 1.0


@pytest.mark.asyncio
async def test_reported_usage_wins_over_depth_price():
    ledger, _, _ = await _run_search("advanced", {"results": [], "usage": {"credits": 1}})
    assert ledger.events[0].meta["credits_used"] == 1.0
