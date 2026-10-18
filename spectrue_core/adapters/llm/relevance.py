# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Semantic gating sensors (M66 restored in M145 Phase R).

Builds the typed Decisions questions for the gates and reads them through
`read_sensor`. Readings are probabilities only; deterministic code in
`use_cases/retrieval/decision_gates.py` fuses them with the step's existing
signal and decides (Constitution II). Question text lives in
`agents/locales/*.yml` under `prompts.decisions` (all 8 languages).
"""

from __future__ import annotations

from typing import Any

from .base_skill import BaseSkill
from spectrue_core.adapters.llm.decision_gates import SensorReading, read_sensor
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.llm.decisions import DecisionQuestion

TOP_SNIPPETS = 3
MAX_CLAIMS_PER_URL = 5


def claim_text(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("normalized_text") or c.get("text") or "")
    return str(getattr(c, "normalized_text", None) or getattr(c, "text", "") or "")


def _snippets_text(results: list[Any]) -> str:
    lines = []
    for res in results[:TOP_SNIPPETS]:
        get = res.get if isinstance(res, dict) else (lambda k, _r=res: getattr(_r, k, None))
        lines.append(f"Source: {get('title') or 'Unknown'}\nSnippet: {get('snippet') or get('content') or ''}\n---")
    return "\n".join(lines)


def _p(lang: str, key: str) -> str:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.{key}")


class RelevanceSkill(BaseSkill):
    """Decisions sensors for semantic gating (G1–G4)."""

    async def read_evidence(
        self, claims: list[Any], *, title: str, snippet: str, gate: str, lang: str = "en",
        trace_meta: dict[str, Any] | None = None,
    ) -> SensorReading | None:
        """One predicate per claim (`c0`, `c1`, ...): does this result hold evidence for it?

        Shared by G3 (`extract`, per URL before extract) and G1
        (`search_relevance`, per result before rerank).
        """
        claims = [c for c in claims if claim_text(c)][:MAX_CLAIMS_PER_URL]
        if not claims:
            return None
        instr = _p(lang, "extract.instructions")
        questions = [DecisionQuestion.predicate(f"c{i}", f"{instr}\nClaim {i}: {claim_text(c)}")
                     for i, c in enumerate(claims)]
        return await read_sensor(
            gate=gate, cfg=self.runtime.decisions, llm_client=self.llm_client,
            input_text=f"Search result title: {title}\nSnippet: {snippet}",
            questions=questions, trace_meta={"locale": lang, "claims": len(claims), **(trace_meta or {})},
        )

    async def read_sufficiency(
        self, claim: Any, sources: list[Any], *, lang: str = "en", claim_id: str | None = None
    ) -> SensorReading | None:
        """Predicate `sufficient`: do the sources found so far suffice to judge the claim? (G4)"""
        return await read_sensor(
            gate="search_stop", cfg=self.runtime.decisions, llm_client=self.llm_client,
            input_text=f"Claim: {claim_text(claim)}\n\nSources found so far:\n{_snippets_text(sources)}",
            questions=[DecisionQuestion.predicate("sufficient", _p(lang, "search_stop.instructions"))],
            trace_meta={"claim_id": claim_id, "locale": lang, "n_sources": len(sources)},
        )

    async def read_verifiability(
        self, claim: Any, *, lang: str = "en", claim_id: str | None = None
    ) -> SensorReading | None:
        """Predicate `checkable`: is this a checkable factual claim? (G2)"""
        return await read_sensor(
            gate="verifiability", cfg=self.runtime.decisions, llm_client=self.llm_client,
            input_text=f"Claim: {claim_text(claim)}",
            questions=[DecisionQuestion.predicate("checkable", _p(lang, "verifiability.instructions"))],
            trace_meta={"claim_id": claim_id, "locale": lang},
        )
