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
Deterministic tests for verdict_ready_for_claim.
Only contract fields are used: claim_id, stance, evidence_tier, quote_matches, source_type.
"""

from spectrue_core.verification.sufficiency import verdict_ready_for_claim


def test_ready_with_quote_matches() -> None:
    sources = [
        {
            "claim_id": "c1",
            "stance": "SUPPORT",
            "quote_matches": True,
            "evidence_tier": "D",
        }
    ]
    ready, stats = verdict_ready_for_claim(sources, claim_id="c1")
    assert ready is True
    assert stats["anchors"] == 1


def test_ready_with_strong_tier_no_quote() -> None:
    sources = [
        {
            "claim_id": "c1",
            "stance": "REFUTE",
            "quote_matches": False,
            "evidence_tier": "A",
        }
    ]
    ready, stats = verdict_ready_for_claim(sources, claim_id="c1")
    assert ready is False
    assert stats["anchors"] == 0


def test_not_ready_without_quote_or_strong_tier() -> None:
    sources = [
        {
            "claim_id": "c1",
            "stance": "SUPPORT",
            "quote_matches": False,
            "evidence_tier": "B",
        }
    ]
    ready, stats = verdict_ready_for_claim(sources, claim_id="c1")
    assert ready is False
    assert stats["anchors"] == 0


def test_not_ready_with_mismatched_claim_id() -> None:
    sources = [
        {
            "claim_id": "c2",
            "stance": "SUPPORT",
            "quote_matches": True,
            "evidence_tier": "A",
        }
    ]
    ready, stats = verdict_ready_for_claim(sources, claim_id="c1")
    assert ready is False
    assert stats["matched_claim_id"] == 0


def test_not_ready_without_quote_matches() -> None:
    sources = [
        {
            "claim_id": "c1",
            "stance": "SUPPORT",
            "quote_matches": False,
            "source_type": "primary",
        }
    ]
    ready, stats = verdict_ready_for_claim(sources, claim_id="c1")
    assert ready is False
    assert stats["quote_matches"] == 0
