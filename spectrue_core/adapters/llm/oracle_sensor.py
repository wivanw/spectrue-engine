# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decisions sensor for fact-check database matching (M147 T025).

One request per oracle lookup: which candidate fact-check (if any) is about the user's claim,
how closely it matches (score 1-5), what that fact-check concluded, how true it rates the claim
(score 1-5) and how harmful the claim is if believed (score 1-5). Candidate texts are data in
the choice descriptions; question text: `prompts.decisions.oracle.*` in all 8 locales.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.decision_gates import SensorReading, read_sensor
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.llm.decisions import DecisionQuestion
from spectrue_core.runtime_config import DecisionsConfig

STATUSES = ("confirmed", "refuted", "mixed")


def _p(lang: str, key: str) -> Any:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.oracle.{key}")


def candidate_label(c: dict) -> str:
    return (f"\"{str(c.get('claim_text', ''))[:300]}\" — rating: {c.get('rating', 'unknown')}; "
            f"publisher: {c.get('publisher', 'unknown')}; {str(c.get('title', ''))[:200]}")


def oracle_questions(lang: str, candidates: list[dict]) -> list[DecisionQuestion]:
    choices = {str(i): candidate_label(c) for i, c in enumerate(candidates)}
    choices["none"] = _p(lang, "best_none")
    return [
        DecisionQuestion.choice("best", _p(lang, "best"), choices),
        DecisionQuestion.score("match", _p(lang, "match"), [(str(i + 1), d) for i, d in enumerate(_p(lang, "match_levels"))]),
        DecisionQuestion.choice("status", _p(lang, "status"), {s: _p(lang, f"status_{s}") for s in STATUSES}),
        DecisionQuestion.score("rating", _p(lang, "rating"), [(str(i + 1), d) for i, d in enumerate(_p(lang, "rating_levels"))]),
        DecisionQuestion.score("harm", get_prompt((lang or "en")[:2], "prompts.decisions.metadata.harm"),
                               [(str(i + 1), d) for i, d in enumerate(get_prompt((lang or "en")[:2], "prompts.decisions.metadata.harm_levels"))]),
    ]


async def read_oracle(llm_client: Any, decisions: DecisionsConfig, *, user_claim: str, candidates: list[dict],
                      lang: str) -> SensorReading | None:
    return await read_sensor(
        gate="oracle", cfg=decisions, llm_client=llm_client,
        input_text=f"User's claim: {user_claim[:1500]}",
        questions=oracle_questions(lang, candidates), trace_meta={"candidates": len(candidates), "locale": lang},
        independent=True,
    )
