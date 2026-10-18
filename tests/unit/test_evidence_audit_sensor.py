# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""M147 T021: evidence audit from the Decisions sensor. No network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectrue_core.adapters.llm.evidence_audit import EvidenceAuditSkill
from spectrue_core.adapters.llm.evidence_audit_sensor import FIELDS, audit_questions
from spectrue_core.llm.decisions import DecisionAnswer, DecisionResult
from spectrue_core.runtime_config import DecisionsConfig

FRAME = SimpleNamespace(claim_id="c1", claim_text="VK reported a hit on a data center.", claim_language="uk")
EV = SimpleNamespace(evidence_id="e1", source_id="s1", url="https://a/vk", title="VK", snippet="VK said...",
                     quote="VK said its data center was hit", source_tier="B")


def _reading_result(**choices):
    return DecisionResult(answers={f: DecisionAnswer(name=f, type="choice", choice=v, confidence=0.9,
                                                     probabilities={v: 0.9}) for f, v in choices.items()},
                          model="m", usage={})


ALL = dict(stance="support", directness="direct", specificity="high", quote_integrity="ok", novelty_vs_copy="original")


def test_questions_in_every_locale():
    for lang in ("en", "uk", "ru", "de", "es", "fr", "ja", "zh"):
        qs = audit_questions(lang)
        assert [q.name for q in qs] == list(FIELDS)
        assert all("Prompt key" not in c["description"] for q in qs for c in q.payload["choices"])


@pytest.mark.asyncio
async def test_on_builds_the_audit_without_an_llm_call():
    llm = MagicMock(); llm.call_decision = AsyncMock(return_value=_reading_result(**ALL)); llm.call_structured = AsyncMock()
    with patch("spectrue_core.adapters.llm.decision_gates.Trace"), patch("spectrue_core.adapters.llm.evidence_audit.Trace"):
        audit = await EvidenceAuditSkill(llm, DecisionsConfig(gate_modes={"evidence_audit": "on"})).audit(FRAME, EV)
    assert not llm.call_structured.called
    assert (audit.stance, audit.directness, audit.specificity) == ("support", "direct", "high")
    assert audit.source_id == "s1" and audit.dependency_hints == [] and audit.audit_confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_on_refused_field_becomes_cannot_tell():
    partial = {k: v for k, v in ALL.items() if k != "specificity"}
    llm = MagicMock(); llm.call_decision = AsyncMock(return_value=_reading_result(**partial))
    with patch("spectrue_core.adapters.llm.decision_gates.Trace"), patch("spectrue_core.adapters.llm.evidence_audit.Trace"):
        audit = await EvidenceAuditSkill(llm, DecisionsConfig(gate_modes={"evidence_audit": "on"})).audit(FRAME, EV)
    assert audit.specificity == "low"


@pytest.mark.asyncio
async def test_off_uses_the_llm_audit_only():
    llm = MagicMock(); llm.call_decision = AsyncMock()
    llm.call_structured = AsyncMock(return_value={"claim_id": "c1", "evidence_id": "e1", "source_id": "s1",
                                                  "dependency_hints": [], "extraction_confidence": 0.8,
                                                  "audit_confidence": 0.8, **ALL})
    with patch("spectrue_core.adapters.llm.evidence_audit.Trace"):
        audit = await EvidenceAuditSkill(llm, DecisionsConfig()).audit(FRAME, EV)
    assert audit.stance == "support" and not llm.call_decision.called
