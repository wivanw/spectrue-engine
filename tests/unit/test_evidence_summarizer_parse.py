# Copyright (C) 2025 Spectrue Contributors
# Tests for evidence summarizer parsing (empty refuting_evidence items, schema tolerance).

from unittest.mock import MagicMock

from spectrue_core.adapters.llm.evidence_summarizer import EvidenceSummarizerSkill


def test_parse_response_tolerates_empty_refuting_evidence_items():
    """LLM sometimes returns refuting_evidence: [{}]; schema allows it and parser ignores empty items."""
    skill = EvidenceSummarizerSkill(llm_client=MagicMock())
    response = {
        "supporting_evidence": [
            {"evidence_id": "a410a3e96299", "reason": "Directly states piezoelectricity."}
        ],
        "refuting_evidence": [{}],
        "contextual_evidence": [
            {"evidence_id": "46a496d8dffc", "reason": "Discusses ultrasonic piezoelectric ceramics."}
        ],
        "evidence_gaps": ["Direct primary-source evidence."],
        "conflicts_present": False,
    }
    summary = skill._parse_response(response, "c6")
    assert len(summary.supporting_evidence) == 1
    assert summary.supporting_evidence[0].evidence_id == "a410a3e96299"
    assert len(summary.refuting_evidence) == 0
    assert len(summary.contextual_evidence) == 1
    assert summary.contextual_evidence[0].evidence_id == "46a496d8dffc"
