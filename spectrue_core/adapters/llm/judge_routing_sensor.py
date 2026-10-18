# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decisions sensor that picks the judge tier from a difficulty score (M147 T026).

Replaces the hand-weighted complexity / ambiguity / risk mix in `judge_model_routing`. One request
per claim: difficulty 1-5 of judging the claim from its evidence. Levels 1-2 allow the cheap tier,
3 the mid tier, 4-5 require the high tier; the cheapest allowed model by expected credits is then
chosen as before. Question text: `prompts.decisions.judge_routing.*` in all 8 locales.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.decision_gates import gate_mode, read_sensor
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.llm.decisions import DecisionQuestion
from spectrue_core.runtime_config import DecisionsConfig

_TIER_BY_LEVEL = {1: "cheap", 2: "cheap", 3: "mid", 4: "high", 5: "high"}


def _p(lang: str, key: str) -> Any:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.judge_routing.{key}")


def routing_questions(lang: str) -> list[DecisionQuestion]:
    return [DecisionQuestion.score("difficulty", _p(lang, "difficulty"),
                                   [(str(i + 1), d) for i, d in enumerate(_p(lang, "levels"))])]


def evidence_digest(evidence: list, limit: int = 8) -> str:
    rows = []
    for e in (evidence or [])[:limit]:
        if isinstance(e, dict):
            rows.append(f"- [{e.get('stance') or 'unclear'}] {str(e.get('quote') or e.get('snippet') or '')[:240]}")
    return "\n".join(rows)


async def required_capability(llm_client: Any, decisions: DecisionsConfig | None, *, claim_text: str,
                              evidence: list, claim_id: str, lang: str = "en") -> str | None:
    """'cheap' | 'mid' | 'high', or None (gate off / no reading: the hand-weighted rule applies)."""
    if decisions is None or gate_mode(decisions, "judge_routing") != "on":
        return None
    reading = await read_sensor(
        gate="judge_routing", cfg=decisions, llm_client=llm_client,
        input_text=f"Claim: {claim_text}\n\nEvidence:\n{evidence_digest(evidence)}"[: decisions.max_input_chars],
        questions=routing_questions(lang), trace_meta={"claim_id": claim_id},
    )
    a = reading.result.answers.get("difficulty") if reading else None
    if a is None or a.score is None:
        return None
    return _TIER_BY_LEVEL[min(5, max(1, int(round(a.score)) + 1))]
