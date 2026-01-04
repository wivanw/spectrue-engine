# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
DAG Pipeline Execution Engine

Extends the linear Pipeline to support step dependencies and parallel execution.

Key Components:
- StepNode: Step wrapper with dependency declarations
- DAGPipeline: Executor with topological sort and parallel execution

Usage:
    from spectrue_core.pipeline.dag import DAGPipeline, StepNode

    pipeline = DAGPipeline(
        mode=NORMAL_MODE,
        nodes=[
            StepNode(step=MeteringSetupStep(config)),
            StepNode(step=ExtractClaimsStep(agent), depends_on=["metering_setup"]),
            StepNode(step=SearchFlowStep(...), depends_on=["extract_claims"]),
        ]
    )
    result = await pipeline.run(initial_ctx)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.constants import (
    DAG_EXECUTION_STATE_KEY,
    DAG_EXECUTION_SUMMARY_KEY,
)
from spectrue_core.pipeline.execution_state import (
    DAGExecutionState,
    LayerExecutionState,
)
from spectrue_core.pipeline.mode import PipelineMode
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# StepNode: Step with Dependencies
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepNode:
    """
    Wrapper for a Step with dependency declarations.

    Attributes:
        step: The Step to execute
        depends_on: List of step names that must complete before this step
        optional: If True, failures are logged but don't stop pipeline
        skip_if: Callable to check if step should be skipped
    """

    step: Step
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False
    skip_if: Any = None  # Callable[[PipelineContext], bool]

    @property
    def name(self) -> str:
        """Get the step name."""
        return self.step.name


# ─────────────────────────────────────────────────────────────────────────────
# DAGPipeline: Executor with Dependency Resolution
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DAGPipeline:
    """
    Executes steps respecting dependencies with parallel execution.

    Steps without dependencies run in parallel. Steps with dependencies
    wait for all dependencies to complete before executing.

    Attributes:
        mode: Pipeline mode configuration
        nodes: List of StepNodes to execute
        max_parallel: Maximum concurrent step executions (default: 5)
    """

    mode: PipelineMode
    nodes: list[StepNode]
    max_parallel: int = 5
    _node_order: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate DAG structure."""
        self._node_order = {node.name: idx for idx, node in enumerate(self.nodes)}
        self._validate_dag()

    def _validate_dag(self) -> None:
        """Ensure no cycles and all dependencies exist."""
        node_names = {n.name for n in self.nodes}
        node_map = {n.name: n for n in self.nodes}
        dependents: dict[str, set[str]] = {name: set() for name in node_names}
        for node in self.nodes:
            for dep in node.depends_on:
                if dep == node.name:
                    raise ValueError(f"Step '{node.name}' cannot depend on itself")
                if dep not in node_names:
                    raise ValueError(
                        f"Step '{node.name}' depends on unknown step '{dep}'"
                    )
                dependents[dep].add(node.name)
        has_edges = any(node.depends_on for node in self.nodes)
        orphan_nodes = [
            name
            for name, node in node_map.items()
            if not node.depends_on and not dependents.get(name)
        ]
        if has_edges and len(self.nodes) > 1 and orphan_nodes:
            orphan_list = ", ".join(sorted(orphan_nodes, key=self._node_order.get))
            raise ValueError(f"DAG has orphan steps with no dependencies: {orphan_list}")
        # Check for cycles using DFS
        self._check_cycles()

    def _check_cycles(self) -> None:
        """Detect cycles in the dependency graph."""
        node_map = {n.name: n for n in self.nodes}
        visited: set[str] = set()
        stack: list[str] = []
        stack_set: set[str] = set()

        def dfs(name: str) -> None:
            visited.add(name)
            stack.append(name)
            stack_set.add(name)
            node = node_map.get(name)
            if node:
                for dep in node.depends_on:
                    if dep not in visited:
                        dfs(dep)
                    elif dep in stack_set:
                        cycle_start = stack.index(dep)
                        cycle = stack[cycle_start:] + [dep]
                        cycle_path = " -> ".join(cycle)
                        raise ValueError(f"Cycle detected in DAG: {cycle_path}")
            stack.pop()
            stack_set.remove(name)

        for node in sorted(self.nodes, key=lambda n: self._node_order.get(n.name, 0)):
            if node.name not in visited:
                dfs(node.name)

    def _topological_sort(self) -> list[list[StepNode]]:
        """
        Sort nodes into execution layers.

        Returns list of layers, where each layer can be executed in parallel.
        """
        node_map = {n.name: n for n in self.nodes}
        in_degree = {n.name: len(n.depends_on) for n in self.nodes}
        layers: list[list[StepNode]] = []

        remaining = set(node_map.keys())

        while remaining:
            # Find all nodes with in_degree 0
            ready = [name for name in remaining if in_degree[name] == 0]
            if not ready:
                blocked = {
                    name: sorted(node_map[name].depends_on)
                    for name in sorted(remaining, key=self._node_order.get)
                }
                raise ValueError(
                    "Cycle detected in DAG. Remaining nodes blocked by dependencies: "
                    f"{blocked}"
                )

            ready_sorted = sorted(ready, key=self._node_order.get)
            layer = [node_map[name] for name in ready_sorted]
            layers.append(layer)

            # Remove ready nodes and update in_degrees
            for name in ready_sorted:
                remaining.remove(name)
                # Update in_degrees for dependent nodes
                for other_name in remaining:
                    other_node = node_map[other_name]
                    if name in other_node.depends_on:
                        in_degree[other_name] -= 1

        return layers

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """
        Execute all steps respecting dependencies.

        Steps in the same layer execute in parallel.

        Args:
            ctx: Initial pipeline context

        Returns:
            Final context after all steps complete

        Raises:
            PipelineViolation: If invariant check fails
            PipelineExecutionError: If step execution fails
        """
        trace = ctx.trace

        if trace:
            Trace.event(
                "dag_pipeline_start",
                {
                    "mode": self.mode.name,
                    "node_count": len(self.nodes),
                    "node_names": [n.name for n in self.nodes],
                },
            )

        layers = self._topological_sort()
        dag_state = DAGExecutionState(started_at=time.time())
        layer_states: list[LayerExecutionState] = []
        for layer_idx, layer in enumerate(layers):
            layer_state = LayerExecutionState(
                index=layer_idx,
                steps=[node.name for node in layer],
            )
            layer_states.append(layer_state)
            for node in layer:
                dag_state.ensure_step(
                    node.name,
                    depends_on=node.depends_on,
                    optional=node.optional,
                    layer=layer_idx,
                )
        dag_state.layers = layer_states
        dag_state.ordered_steps = [node.name for layer in layers for node in layer]

        current_ctx = ctx.set_extra(DAG_EXECUTION_STATE_KEY, dag_state)
        completed_results: dict[str, PipelineContext] = {}

        for layer_idx, layer in enumerate(layers):
            if trace:
                Trace.event(
                    "dag_layer_start",
                    {
                        "layer_idx": layer_idx,
                        "steps": [n.name for n in layer],
                    },
                )
            layer_state = dag_state.layers[layer_idx]
            layer_state.mark_started(timestamp=time.time())

            # Filter nodes that should run
            nodes_to_run = []
            for node in layer:
                if node.skip_if and node.skip_if(current_ctx):
                    Trace.event("dag_step_skipped", {"step": node.name, "reason": "skip_if"})
                    step_state = dag_state.ensure_step(
                        node.name,
                        depends_on=node.depends_on,
                        optional=node.optional,
                        layer=layer_idx,
                    )
                    step_state.mark_skipped(timestamp=time.time(), reason="skip_if")
                    continue
                nodes_to_run.append(node)

            if not nodes_to_run:
                layer_state.mark_completed(timestamp=time.time())
                continue

            # Execute layer in parallel (with semaphore limit)
            semaphore = asyncio.Semaphore(self.max_parallel)

            async def run_node(node: StepNode) -> tuple[str, PipelineContext | Exception]:
                async with semaphore:
                    try:
                        step_state = dag_state.ensure_step(
                            node.name,
                            depends_on=node.depends_on,
                            optional=node.optional,
                            layer=layer_idx,
                        )
                        step_state.mark_running(timestamp=time.time())
                        if trace:
                            Trace.event("dag_step_start", {"step": node.name})
                        result = await node.step.run(current_ctx)
                        step_state.mark_succeeded(timestamp=time.time())
                        if trace:
                            Trace.event("dag_step_end", {"step": node.name})
                        return node.name, result
                    except Exception as e:
                        step_state = dag_state.ensure_step(
                            node.name,
                            depends_on=node.depends_on,
                            optional=node.optional,
                            layer=layer_idx,
                        )
                        step_state.mark_failed(timestamp=time.time(), error=e)
                        if trace:
                            Trace.event(
                                "dag_step_error",
                                {"step": node.name, "error": str(e)},
                            )
                        if node.optional:
                            logger.warning("[DAG] Optional step '%s' failed: %s", node.name, e)
                            return node.name, current_ctx
                        return node.name, e

            results = await asyncio.gather(
                *[run_node(node) for node in nodes_to_run],
                return_exceptions=False,
            )

            # Process results and merge contexts
            results_by_name = {name: result for name, result in results}
            merge_order = [node.name for node in nodes_to_run]
            for name in merge_order:
                result = results_by_name.get(name)
                if isinstance(result, Exception):
                    raise result
                if result is not None:
                    completed_results[name] = result

            # Merge contexts from this layer
            # Properly merge extras from all parallel step results
            merged_extras = dict(current_ctx.extras) if current_ctx.extras else {}
            merged_claims = current_ctx.claims
            merged_sources = current_ctx.sources
            merged_verdict = current_ctx.verdict
            merged_evidence = current_ctx.evidence

            for name in merge_order:
                result = results_by_name.get(name)
                if isinstance(result, Exception) or result is None:
                    continue
                # Merge extras additively
                if result.extras:
                    merged_extras.update(result.extras)
                # Use latest non-empty values deterministically
                if result.claims:
                    merged_claims = result.claims
                if result.sources:
                    merged_sources = result.sources
                if result.verdict:
                    merged_verdict = result.verdict
                if result.evidence is not None:
                    merged_evidence = result.evidence

            current_ctx = PipelineContext(
                mode=current_ctx.mode,
                claims=merged_claims,
                lang=current_ctx.lang,
                search_type=current_ctx.search_type,
                gpt_model=current_ctx.gpt_model,
                trace=current_ctx.trace,
                sources=merged_sources,
                evidence=merged_evidence,
                verdict=merged_verdict,
                extras=merged_extras,
            )
            layer_state.mark_completed(timestamp=time.time())

        dag_state.completed_at = time.time()
        dag_state_summary = dag_state.to_summary()
        dag_state_dict = dag_state.to_dict()
        current_ctx = current_ctx.set_extra(
            DAG_EXECUTION_SUMMARY_KEY,
            dag_state_summary,
        ).set_extra(
            DAG_EXECUTION_STATE_KEY,
            dag_state_dict,
        )
        final_result = current_ctx.get_extra("final_result")
        if isinstance(final_result, dict):
            final_result = {
                **final_result,
                "dag_execution_summary": dag_state_summary,
                "dag_execution_state": dag_state_dict,
            }
            current_ctx = current_ctx.set_extra("final_result", final_result)

        if trace:
            Trace.event(
                "dag_pipeline_end",
                {
                    "mode": self.mode.name,
                    "completed_steps": list(completed_results.keys()),
                    "execution_summary": dag_state_summary,
                },
            )

        return current_ctx

    def __repr__(self) -> str:
        node_names = [n.name for n in self.nodes]
        return f"DAGPipeline(mode={self.mode.name}, nodes={node_names})"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Build Normal Pipeline DAG
# ─────────────────────────────────────────────────────────────────────────────


def build_normal_pipeline_dag(
    *,
    config: Any,
    agent: Any,
    search_mgr: Any,
    pipeline: Any,  # ValidationPipeline
) -> DAGPipeline:
    """
    Build the normal mode pipeline as a DAG.

    Normal Pipeline Flow:
        MeteringSetup ──┐
                        ├──▶ PrepareInput ──▶ ExtractClaims ──▶ OracleFlow
                        │                                          │
                        │                                          ▼
                        │         TargetSelection ◀── ClaimGraph ──┘
                        │                │
                        │                ▼
                        │          SearchFlow ──▶ EvidenceFlow ──▶ ResultAssembly
                        └───────────────────────────────────────────────────────┘
    """
    from spectrue_core.pipeline.steps import (
        MeteringSetupStep,
        PrepareInputStep,
        ExtractClaimsStep,
        ClaimGraphStep,
        TargetSelectionStep,
        SearchFlowStep,
        EvidenceFlowStep,
        OracleFlowStep,
        ResultAssemblyStep,
    )
    from spectrue_core.pipeline.steps.invariants import (
        AssertNonEmptyClaimsStep,
        AssertSingleClaimStep,
    )
    from spectrue_core.pipeline.mode import NORMAL_MODE

    nodes = [
        # Infrastructure
        StepNode(step=MeteringSetupStep(config=config)),

        # Invariants (after metering, before claims)
        StepNode(
            step=AssertNonEmptyClaimsStep(),
            depends_on=["extract_claims"],
        ),

        # Input preparation
        StepNode(
            step=PrepareInputStep(pipeline=pipeline),
            depends_on=["assert_non_empty_claims"],
        ),

        # Claim extraction
        StepNode(
            step=ExtractClaimsStep(agent=agent),
            depends_on=["prepare_input"],
        ),

        # Normal mode: single claim assertion
        StepNode(
            step=AssertSingleClaimStep(),
            depends_on=["extract_claims"],
        ),

        # Oracle fast path (optional)
        StepNode(
            step=OracleFlowStep(pipeline=pipeline),
            depends_on=["assert_single_claim"],
            optional=True,
        ),

        # Claim graph (optional, can run in parallel with oracle)
        StepNode(
            step=ClaimGraphStep(config=config),
            depends_on=["extract_claims"],
            optional=True,
        ),

        # Target selection
        StepNode(
            step=TargetSelectionStep(),
            depends_on=["claim_graph", "oracle_flow"],
        ),

        # Search retrieval
        StepNode(
            step=SearchFlowStep(config=config, search_mgr=search_mgr, agent=agent),
            depends_on=["target_selection"],
        ),

        # Evidence scoring
        StepNode(
            step=EvidenceFlowStep(agent=agent, search_mgr=search_mgr),
            depends_on=["search_flow"],
        ),

        # Final assembly
        StepNode(
            step=ResultAssemblyStep(),
            depends_on=["evidence_flow"],
        ),
    ]

    return DAGPipeline(mode=NORMAL_MODE, nodes=nodes)
