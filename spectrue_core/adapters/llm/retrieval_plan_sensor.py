# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decisions sensor that replaces call B (claim_retrieval_plan), M147 T022 / M146 T033.

One request per claim: the metadata questions (M146 T031) plus the classification fields
call B still owned (evidence_need, search_method, needs_primary, structure_type). The text
fields call B produced (search queries) come from call A. Question text:
`prompts.decisions.metadata.*` and `prompts.decisions.plan.*` in all 8 locales.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.decision_gates import SensorReading, read_sensor
from spectrue_core.adapters.llm.metadata_sensor import metadata_questions
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.llm.decisions import DecisionQuestion
from spectrue_core.runtime_config import DecisionsConfig

EVIDENCE_NEEDS = ("official_stats", "empirical_study", "guideline", "news_report", "expert_opinion",
                  "definition", "anecdotal", "unknown")
SEARCH_METHODS = ("general_search", "news", "academic")
STRUCTURE_TYPES = ("empirical_numeric", "event", "causal", "attribution", "definition", "policy_plan",
                   "forecast", "meta_scientific", "other")


def _p(lang: str, key: str) -> Any:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.plan.{key}")


def plan_questions(lang: str) -> list[DecisionQuestion]:
    return metadata_questions(lang) + [
        DecisionQuestion.choice("evidence_need", _p(lang, "evidence_need"), {v: _p(lang, f"need_{v}") for v in EVIDENCE_NEEDS}),
        DecisionQuestion.choice("search_method", _p(lang, "search_method"), {v: _p(lang, f"method_{v}") for v in SEARCH_METHODS}),
        DecisionQuestion.predicate("needs_primary", _p(lang, "needs_primary")),
        DecisionQuestion.choice("structure_type", _p(lang, "structure_type"), {v: _p(lang, f"type_{v}") for v in STRUCTURE_TYPES}),
    ]


async def read_retrieval_plan(llm_client: Any, decisions: DecisionsConfig, *, claim_text: str, window: str,
                              lang: str, claim_id: str) -> SensorReading | None:
    return await read_sensor(
        gate="retrieval_plan", cfg=decisions, llm_client=llm_client,
        input_text=f"Claim: {claim_text}\n\nSurrounding text:\n{window}"[: decisions.max_input_chars],
        questions=plan_questions(lang), trace_meta={"claim_id": claim_id, "locale": lang}, independent=True,
    )
