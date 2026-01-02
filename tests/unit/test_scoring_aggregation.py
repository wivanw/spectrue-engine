# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json
from pathlib import Path

from spectrue_core.verification.scoring.scoring_aggregation import aggregate_claim_verdict


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "scoring"


def _load_fixture(name: str) -> dict:
    data = json.loads((FIXTURES_DIR / name).read_text())
    return data


def test_tier_a_refute_dominates_many_low_support() -> None:
    fixture = _load_fixture("tierA_refute_vs_many_low_support.json")
    evidence_pack = fixture["evidence_pack"]
    result = aggregate_claim_verdict(evidence_pack, claim_id=fixture["claim_id"])

    assert result["verdict"] == "ambiguous"
    assert 0.35 <= result["verdict_score"] <= 0.65


def test_strong_conflict_same_tier_is_ambiguous() -> None:
    fixture = _load_fixture("strong_conflict_same_tier.json")
    evidence_pack = fixture["evidence_pack"]
    result = aggregate_claim_verdict(evidence_pack, claim_id=fixture["claim_id"])

    assert result["verdict"] == "ambiguous"
    assert 0.35 <= result["verdict_score"] <= 0.65


def test_tier_does_not_affect_verdict_score() -> None:
    evidence_pack_a = {
        "items": [
            {
                "claim_id": "c1",
                "stance": "SUPPORT",
                "quote": "Direct quote.",
                "relevance": 0.7,
                "tier": "A",
            }
        ],
        "stats": {"domain_diversity": 1},
    }
    evidence_pack_d = {
        "items": [
            {
                "claim_id": "c1",
                "stance": "SUPPORT",
                "quote": "Direct quote.",
                "relevance": 0.7,
                "tier": "D",
            }
        ],
        "stats": {"domain_diversity": 1},
    }

    result_a = aggregate_claim_verdict(evidence_pack_a, claim_id="c1")
    result_d = aggregate_claim_verdict(evidence_pack_d, claim_id="c1")

    assert result_a["verdict"] == result_d["verdict"]
    assert result_a["verdict_score"] == result_d["verdict_score"]
