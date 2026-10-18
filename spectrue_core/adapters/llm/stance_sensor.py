# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decisions sensor for evidence stance (M147 T020).

One Decisions request per (source, claim) pair with a single `choice` question:
support / refute / context / irrelevant, answered with probabilities. Readings are
probabilities only; `use_cases/evidence/stance_sensor.py` decides what to apply
(Constitution II). Question text: `prompts.decisions.stance.*` in all 8 locales.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.decision_gates import SensorReading, read_sensor
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.llm.decisions import DecisionQuestion
from spectrue_core.runtime_config import DecisionsConfig

STANCES = ("support", "refute", "context", "irrelevant")


def _p(lang: str, key: str) -> Any:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.stance.{key}")


def stance_questions(lang: str) -> list[DecisionQuestion]:
    return [DecisionQuestion.choice("stance", _p(lang, "instructions"), {s: _p(lang, s) for s in STANCES})]


async def read_stance(
    llm_client: Any, decisions: DecisionsConfig, *, claim_text: str, source_text: str, lang: str,
    claim_id: str, url: str,
) -> SensorReading | None:
    budget = max(200, decisions.max_input_chars - len(claim_text) - 40)
    return await read_sensor(
        gate="stance", cfg=decisions, llm_client=llm_client,
        input_text=f"Claim: {claim_text}\n\nSource:\n{source_text[:budget]}",
        questions=stance_questions(lang), trace_meta={"claim_id": claim_id, "url": url[:160], "locale": lang},
    )
