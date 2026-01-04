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

from unittest.mock import MagicMock

import pytest

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.factory import PipelineFactory
from spectrue_core.pipeline.mode import NORMAL_MODE
from spectrue_core.pipeline.steps.result_assembly import AssembleStandardResultStep
from spectrue_core.pipeline.steps.metering_setup import METERING_SETUP_STEP_NAME


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.runtime = MagicMock()
    config.runtime.features = MagicMock()
    return config


@pytest.fixture
def pipeline_factory():
    return PipelineFactory(search_mgr=MagicMock(), agent=MagicMock())


def test_standard_graph_invariants(mock_config, pipeline_factory):
    pipeline = pipeline_factory.build("normal", config=mock_config)
    names = [node.name for node in pipeline.nodes]

    assert METERING_SETUP_STEP_NAME in names
    assert "judge_standard" in names
    assert "judge_claims" not in names
    assert "assert_standard_result_keys" in names


@pytest.mark.asyncio
async def test_standard_result_contract_fields():
    verdict = {
        "judge_mode": "standard",
        "rgba": [0.1, 0.2, 0.3, 0.4],
        "rationale": "Example rationale",
        "sources": [{"url": "https://example.com", "title": "Example"}],
        "anchor_claim": {"id": "c1", "text": "Example claim"},
        "claim_verdicts": [
            {
                "claim_id": "c1",
                "text": "Example claim",
                "rgba": [0.1, 0.2, 0.3, 0.4],
                "rationale": "Example rationale",
            }
        ],
    }
    ctx = PipelineContext(mode=NORMAL_MODE, claims=[{"id": "c1", "text": "Example claim"}])
    ctx = ctx.with_update(verdict=verdict, sources=verdict["sources"])
    ctx = ctx.set_extra("prepared_fact", "Prepared text")
    ctx = ctx.set_extra("cost_summary", {"credits_used": 2})

    result_ctx = await AssembleStandardResultStep().run(ctx)
    final_result = result_ctx.get_extra("final_result")

    assert final_result["judge_mode"] == "standard"
    assert final_result["text"] == "Prepared text"
    assert isinstance(final_result["details"], list)
    assert final_result["anchor_claim"]["id"] == "c1"
    assert final_result["rgba"] == [0.1, 0.2, 0.3, 0.4]
    assert "deep_analysis" not in final_result
    assert "cost_summary" in final_result
    assert final_result.get("credits") == 2
