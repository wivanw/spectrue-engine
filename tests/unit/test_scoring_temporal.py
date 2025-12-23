import json
from pathlib import Path

from spectrue_core.verification.scoring_aggregation import aggregate_claim_verdict


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "scoring"


def _load_fixture(name: str) -> dict:
    data = json.loads((FIXTURES_DIR / name).read_text())
    return data


def test_context_only_does_not_verify() -> None:
    fixture = _load_fixture("context_only_should_not_verify.json")
    evidence_pack = fixture["evidence_pack"]
    result = aggregate_claim_verdict(evidence_pack, claim_id=fixture["claim_id"])

    assert result["verdict"] != "verified"


def test_temporal_mismatch_prevents_verification() -> None:
    fixture = _load_fixture("temporal_mismatch.json")
    evidence_pack = fixture["evidence_pack"]
    temporality = fixture["claim"].get("temporality")
    result = aggregate_claim_verdict(
        evidence_pack,
        claim_id=fixture["claim_id"],
        temporality=temporality,
    )

    assert result["verdict"] != "verified"
    assert result["verdict_score"] < 0.8
