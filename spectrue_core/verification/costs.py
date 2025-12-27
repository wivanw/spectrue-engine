"""
Cost constants for verification operations.
"""

# Model costs (credits per claim analysis)
MODEL_COSTS = {
    "gpt-5-nano": 5,
    "gpt-5-mini": 20,
    "gpt-5.2": 100,
}

# Search costs (credits per search operation)
SEARCH_COSTS = {
    "basic": 80,
    "advanced": 160,
}


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
