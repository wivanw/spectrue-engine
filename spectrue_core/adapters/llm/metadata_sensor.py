# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Decisions sensor for claim metadata (M146 T031).

One Decisions request per claim with independent typed questions. Readings are
probabilities only; `use_cases/claims/metadata_sensor.py` decides which fields
to take (Constitution II). Question text: `prompts.decisions.metadata.*` in all
8 locales.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.adapters.llm.decision_gates import SensorReading, read_sensor
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.llm.decisions import DecisionQuestion
from spectrue_core.runtime_config import DecisionsConfig

VERIFICATION_TARGETS = ("reality", "attribution", "existence", "none")
CLAIM_ROLES = ("core", "support", "context", "counterclaim", "attribution")


def _p(lang: str, key: str) -> Any:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.metadata.{key}")


def metadata_questions(lang: str) -> list[DecisionQuestion]:
    return [
        DecisionQuestion.choice("verification_target", _p(lang, "verification_target"),
                                {v: _p(lang, f"vt_{v}") for v in VERIFICATION_TARGETS}),
        DecisionQuestion.choice("claim_role", _p(lang, "claim_role"),
                                {r: _p(lang, f"role_{r}") for r in CLAIM_ROLES}),
        DecisionQuestion.score("harm", _p(lang, "harm"),
                               [(str(i + 1), d) for i, d in enumerate(_p(lang, "harm_levels"))]),
        DecisionQuestion.score("importance", _p(lang, "importance"),
                               [(str(i + 1), d) for i, d in enumerate(_p(lang, "importance_levels"))]),
        DecisionQuestion.predicate("check_oracle", _p(lang, "check_oracle")),
        DecisionQuestion.predicate("time_sensitive", _p(lang, "time_sensitive")),
        DecisionQuestion.predicate("satire", _p(lang, "satire")),
    ]


async def read_claim_metadata(
    llm_client: Any, decisions: DecisionsConfig, *, claim_text: str, window: str, lang: str, claim_id: str
) -> SensorReading | None:
    return await read_sensor(
        gate="metadata", cfg=decisions, llm_client=llm_client,
        input_text=f"Claim: {claim_text}\n\nSurrounding text:\n{window}",
        questions=metadata_questions(lang), trace_meta={"claim_id": claim_id, "locale": lang},
        independent=True,
    )
