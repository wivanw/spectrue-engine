"""Claim graph use cases."""

from __future__ import annotations

from typing import Any

from spectrue_core.pipeline.claim_graph_flow import run_claim_graph_flow


async def build_claim_graph(*, claim_graph: Any, claims: list[dict], runtime_config: Any, progress_callback=None):
    return await run_claim_graph_flow(
        claim_graph,
        claims=claims,
        runtime_config=runtime_config,
        progress_callback=progress_callback,
    )
