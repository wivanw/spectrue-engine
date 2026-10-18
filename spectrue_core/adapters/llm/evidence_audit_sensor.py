# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decisions sensor for the deep-mode evidence audit (M147 T021).

The audit's fields are all categorical (stance, directness, specificity, quote integrity,
novelty), so one Decisions request per (claim, evidence) answers them as independent typed
questions. `dependency_hints` (free text) has no consumer and is left empty. Question text:
`prompts.decisions.evidence_audit.*` in all 8 locales.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.decision_gates import SensorReading, read_sensor
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.llm.decisions import DecisionQuestion
from spectrue_core.runtime_config import DecisionsConfig

FIELDS: dict[str, tuple[str, ...]] = {
    "stance": ("support", "refute", "unclear", "unrelated"),
    "directness": ("direct", "indirect", "tangential"),
    "specificity": ("high", "medium", "low"),
    "quote_integrity": ("ok", "partial", "out_of_context", "not_applicable"),
    "novelty_vs_copy": ("original", "syndicated", "unknown"),
}


def _p(lang: str, key: str) -> Any:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.evidence_audit.{key}")


def audit_questions(lang: str) -> list[DecisionQuestion]:
    return [DecisionQuestion.choice(f, _p(lang, f), {v: _p(lang, f"{f}_{v}") for v in values})
            for f, values in FIELDS.items()]


def audit_input(frame: Any, evidence: Any) -> str:
    return (f"Claim: {frame.claim_text}\n\nSource: {evidence.title or ''} ({evidence.url})\n"
            f"Trust tier: {evidence.source_tier or 'unknown'}\n"
            f"Quote: {evidence.quote or ''}\n\nExcerpt:\n{evidence.snippet or ''}")


async def read_evidence_audit(llm_client: Any, decisions: DecisionsConfig, frame: Any, evidence: Any,
                              lang: str) -> SensorReading | None:
    text = audit_input(frame, evidence)[: decisions.max_input_chars]
    return await read_sensor(
        gate="evidence_audit", cfg=decisions, llm_client=llm_client, input_text=text,
        questions=audit_questions(lang),
        trace_meta={"claim_id": frame.claim_id, "evidence_id": evidence.evidence_id, "locale": lang},
        independent=True,
    )
