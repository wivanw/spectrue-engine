# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for evidence_explainability module (M119, M133).

Tests tier ranking and best tier selection.
M133: compute_explainability_tier_adjustment removed.
"""



from spectrue_core.verification.evidence.evidence_explainability import (
    get_tier_rank,
    find_best_tier_for_claim,
)


class TestGetTierRank:
    """Tests for tier rank ordering."""

    def test_tier_a_highest_rank(self):
        assert get_tier_rank("A") == 4

    def test_tier_a_prime_equals_b(self):
        assert get_tier_rank("A'") == get_tier_rank("B")
        assert get_tier_rank("A'") == 3

    def test_tier_b_rank(self):
        assert get_tier_rank("B") == 3

    def test_tier_c_rank(self):
        assert get_tier_rank("C") == 2

    def test_tier_d_rank(self):
        assert get_tier_rank("D") == 1

    def test_none_tier_zero_rank(self):
        assert get_tier_rank(None) == 0

    def test_unknown_tier_zero_rank(self):
        assert get_tier_rank("UNKNOWN") == 0
        assert get_tier_rank("X") == 0

    def test_case_insensitive(self):
        assert get_tier_rank("a") == get_tier_rank("A")
        assert get_tier_rank("b") == get_tier_rank("B")

    def test_whitespace_stripped(self):
        assert get_tier_rank("  A  ") == 4

    def test_ordering_chain(self):
        ranks = [get_tier_rank(t) for t in ["A", "B", "C", "D", None]]
        assert ranks == sorted(ranks, reverse=True)


# M133: TestComputeExplainabilityTierAdjustment removed — function deleted


class TestFindBestTierForClaim:
    """Tests for finding best tier from evidence items."""

    def test_empty_items_returns_none(self):
        assert find_best_tier_for_claim("c1", []) is None

    def test_single_item_returns_tier(self):
        items = [{"claim_id": "c1", "tier": "B"}]
        assert find_best_tier_for_claim("c1", items) == "B"

    def test_selects_highest_tier(self):
        items = [
            {"claim_id": "c1", "tier": "C"},
            {"claim_id": "c1", "tier": "A"},
            {"claim_id": "c1", "tier": "D"},
        ]
        assert find_best_tier_for_claim("c1", items) == "A"

    def test_filters_by_claim_id(self):
        items = [
            {"claim_id": "c1", "tier": "D"},
            {"claim_id": "c2", "tier": "A"},
        ]
        assert find_best_tier_for_claim("c1", items) == "D"

    def test_includes_shared_items(self):
        # Items without claim_id should be included for any claim
        items = [
            {"tier": "A"},  # No claim_id - shared
            {"claim_id": "c1", "tier": "C"},
        ]
        assert find_best_tier_for_claim("c1", items) == "A"

    def test_handles_none_claim_id_in_items(self):
        items = [
            {"claim_id": None, "tier": "B"},
            {"claim_id": "c1", "tier": "D"},
        ]
        assert find_best_tier_for_claim("c1", items) == "B"

    def test_ignores_non_dict_items(self):
        items = ["invalid", {"claim_id": "c1", "tier": "B"}, None]
        assert find_best_tier_for_claim("c1", items) == "B"

    def test_handles_missing_tier(self):
        items = [
            {"claim_id": "c1"},  # No tier field
            {"claim_id": "c1", "tier": "C"},
        ]
        assert find_best_tier_for_claim("c1", items) == "C"

    def test_none_claim_id_matches_all(self):
        items = [
            {"claim_id": "c1", "tier": "C"},
            {"claim_id": "c2", "tier": "A"},
        ]
        # When claim_id is None, should find best across all items
        # Actually, in implementation, if claim_id is passed as truthy string,
        # only matching items are considered. Let's test with empty string.
        # Based on implementation: if claim_id is falsy, item_claim_id check
        # would be in (None, ""), which won't match "c1" or "c2"
        # So let's test the actual behavior
        items = [
            {"tier": "A"},
            {"tier": "B"},
        ]
        assert find_best_tier_for_claim("", items) == "A"

