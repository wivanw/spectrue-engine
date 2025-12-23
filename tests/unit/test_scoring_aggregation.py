import json
from pathlib import Path

from spectrue_core.verification.scoring_aggregation import aggregate_claim_verdict


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "scoring"


def _load_fixture(name: str) -> dict:
    data = json.loads((FIXTURES_DIR / name).read_text())
    return data


def test_tier_a_refute_dominates_many_low_support() -> None:
    fixture = _load_fixture("tierA_refute_vs_many_low_support.json")
    evidence_pack = fixture["evidence_pack"]
    result = aggregate_claim_verdict(evidence_pack, claim_id=fixture["claim_id"])

    assert result["verdict"] == "refuted"
    assert result["verdict_score"] <= 0.2


def test_strong_conflict_same_tier_is_ambiguous() -> None:
    fixture = _load_fixture("strong_conflict_same_tier.json")
    evidence_pack = fixture["evidence_pack"]
    result = aggregate_claim_verdict(evidence_pack, claim_id=fixture["claim_id"])

    assert result["verdict"] == "ambiguous"
    assert 0.35 <= result["verdict_score"] <= 0.65
