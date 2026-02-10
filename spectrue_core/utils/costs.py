# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Cost constants and utilities for verification operations.
"""


def summarize_reason_codes(
    reason_codes: list[dict],
    *,
    max_items: int = 8,
) -> list[dict]:
    """
    Aggregate reason codes into sorted summaries by cost then count.
    """
    summary: dict[str, dict] = {}
    for rc in reason_codes:
        code = rc.get("code")
        if not code:
            continue
        entry = summary.setdefault(
            code,
            {"code": code, "count": 0, "sc_cost": 0.0, "tc_cost": 0.0},
        )
        entry["count"] += int(rc.get("count", 1) or 1)
        entry["sc_cost"] += float(rc.get("sc_cost", 0.0) or 0.0)
        entry["tc_cost"] += float(rc.get("tc_cost", 0.0) or 0.0)

    ordered = sorted(
        summary.values(),
        key=lambda x: (x.get("sc_cost", 0.0), x.get("count", 0)),
        reverse=True,
    )
    return ordered[:max_items]


def map_stage_costs_to_phases(by_stage_credits: dict[str, float]) -> dict[str, float]:
    """
    Map cost stages to logical phases for ledger reporting.
    """
    phases = {
        "extraction": 0.0,
        "graph": 0.0,
        "query_build": 0.0,
        "retrieval": 0.0,
        "evidence_eval": 0.0,
        "verdict": 0.0,
    }

    for stage, cost in (by_stage_credits or {}).items():
        stage_key = str(stage).lower()
        if "claim_extraction" in stage_key or "article_clean" in stage_key:
            phases["extraction"] += float(cost)
        elif "edge_typing" in stage_key or "claim_graph" in stage_key:
            phases["graph"] += float(cost)
        elif "query_generation" in stage_key:
            phases["query_build"] += float(cost)
        elif "search" in stage_key or "extract" in stage_key:
            phases["retrieval"] += float(cost)
        elif "score_evidence" in stage_key or "analysis" in stage_key:
            phases["evidence_eval"] += float(cost)
        elif "verdict" in stage_key:
            phases["verdict"] += float(cost)
        elif "llm" in stage_key:
            phases["evidence_eval"] += float(cost)

    return phases
