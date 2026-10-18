# Copyright (C) 2025 Ivan Bondarenko
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Decisions sensor for claim-graph edge typing (M147, deep mode).

One request per candidate edge: the relation between two claims as a choice; its probability is
the edge score. Free-text rationale / spans have no consumer beyond logs and are left empty.
Question text: `prompts.decisions.edge.*` in all 8 locales.
"""

from __future__ import annotations

import asyncio
from typing import Any

from spectrue_core.adapters.llm.decision_gates import read_sensor
from spectrue_core.agents.prompts import get_prompt
from spectrue_core.domain.claims.graph.edges import EdgeRelation, TypedEdge
from spectrue_core.llm.decisions import DecisionQuestion
from spectrue_core.runtime_config import DecisionsConfig

RELATIONS = tuple(r.value for r in EdgeRelation)
MAX_CONCURRENT_EDGE_READS = 16


def _p(lang: str, key: str) -> Any:
    return get_prompt((lang or "en")[:2], f"prompts.decisions.edge.{key}")


def edge_questions(lang: str) -> list[DecisionQuestion]:
    return [DecisionQuestion.choice("relation", _p(lang, "instructions"), {r: _p(lang, r) for r in RELATIONS})]


async def type_edges(llm_client: Any, decisions: DecisionsConfig, edges: list, node_map: dict, *,
                     lang: str = "en", max_claim_chars: int = 280) -> list[TypedEdge | None]:
    sem = asyncio.Semaphore(MAX_CONCURRENT_EDGE_READS)

    def text(cid: str) -> str:
        node = node_map.get(cid)
        return str(getattr(node, "text", "") or "")[:max_claim_chars]

    async def one(e) -> TypedEdge | None:
        async with sem:
            reading = await read_sensor(
                gate="edge_typing", cfg=decisions, llm_client=llm_client,
                input_text=f"C1: {text(e.src_id)}\nC2: {text(e.dst_id)}",
                questions=edge_questions(lang), trace_meta={"src": e.src_id, "dst": e.dst_id},
            )
        a = reading.result.answers.get("relation") if reading else None
        if a is None or a.choice not in RELATIONS:
            return None
        return TypedEdge(src_id=e.src_id, dst_id=e.dst_id, relation=EdgeRelation(a.choice),
                         score=float(a.top_confidence or 0.0), rationale_short="", evidence_spans="",
                         cross_topic=getattr(e, "cross_topic", False), same_section=getattr(e, "same_section", False),
                         reason=getattr(e, "reason", "sim"))

    return list(await asyncio.gather(*(one(e) for e in edges)))
