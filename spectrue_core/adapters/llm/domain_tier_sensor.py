# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decisions sensor for the reliability tier of domains missing from the curated lists (M147 T033).

Unknown domains used to be tier D (low reliability) by default. One request per unknown domain
(cached per process) chooses its evidence channel from the domain and the titles it returned.
Question text: `prompts.decisions.domain_tier.*` in all 8 locales.
"""

from __future__ import annotations

import asyncio
from typing import Any

from spectrue_core.adapters.llm.decision_gates import gate_mode, read_sensor
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.llm.decisions import DecisionQuestion
from spectrue_core.runtime_config import DecisionsConfig

CHANNELS = ("authoritative", "reputable_news", "local_media", "social", "low_reliability")


def _p(lang: str, key: str) -> Any:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.domain_tier.{key}")


def tier_questions(lang: str) -> list[DecisionQuestion]:
    return [DecisionQuestion.choice("channel", _p(lang, "instructions"), {c: _p(lang, c) for c in CHANNELS})]


async def resolve_unknown_domains(llm_client: Any, decisions: DecisionsConfig, titles_by_domain: dict[str, list[str]],
                                  *, lang: str = "en") -> int:
    """Classify domains the curated lists do not know; results go to the trusted_sources cache. Returns count."""
    from spectrue_core.tools.trusted_sources import is_known_domain, record_sensor_tier, sensor_tier

    if gate_mode(decisions, "domain_tier") == "off" or llm_client is None:
        return 0
    todo = {d: t for d, t in titles_by_domain.items() if d and not is_known_domain(d) and sensor_tier(d) is None}

    async def one(domain: str, titles: list[str]):
        reading = await read_sensor(
            gate="domain_tier", cfg=decisions, llm_client=llm_client,
            input_text=f"Domain: {domain}\nPage titles from this domain:\n" + "\n".join(f"- {t}" for t in titles[:5]),
            questions=tier_questions(lang), trace_meta={"domain": domain},
        )
        answer = reading.result.answers.get("channel") if reading else None
        if reading is not None and reading.applied and answer is not None and answer.choice in CHANNELS:
            record_sensor_tier(domain, answer.choice)
            return 1
        return 0

    return sum(await asyncio.gather(*(one(d, t) for d, t in todo.items())))
