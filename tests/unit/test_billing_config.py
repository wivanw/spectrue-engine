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

from spectrue_core.billing.config_loader import load_pricing_policy


def test_pricing_config_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SPECTRUE_TAVILY_MULTIPLIER", "12")
    monkeypatch.setenv("SPECTRUE_USD_PER_CREDIT", "0.02")
    policy = load_pricing_policy()
    assert policy.tavily_usd_multiplier == 12.0
    assert policy.usd_per_spectrue_credit == 0.02
