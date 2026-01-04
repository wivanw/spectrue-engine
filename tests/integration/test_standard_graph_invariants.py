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

from spectrue_core.pipeline.factory import PipelineFactory
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
