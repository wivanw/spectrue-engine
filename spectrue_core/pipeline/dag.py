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
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.pipeline.core import PipelineContext, Step
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

    def __post_init__(self) -> None:
        """Validate DAG structure."""
        self._validate_dag()

    def _validate_dag(self) -> None:
        """Ensure no cycles and all dependencies exist."""
        node_names = {n.name for n in self.nodes}
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in node_names:
                    raise ValueError(
                        f"Step '{node.name}' depends on unknown step '{dep}'"
                    )
        # Check for cycles using DFS
        self._check_cycles()

    def _check_cycles(self) -> None:
        """Detect cycles in the dependency graph."""
        visited: set[str] = set()
        rec_stack: set[str] = set()
        node_map = {n.name: n for n in self.nodes}

        def dfs(name: str) -> bool:
            visited.add(name)
            rec_stack.add(name)
            node = node_map.get(name)
            if node:
                for dep in node.depends_on:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            rec_stack.remove(name)
            return False

        for node in self.nodes:
            if node.name not in visited:
                if dfs(node.name):
                    raise ValueError(f"Cycle detected in DAG involving '{node.name}'")

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
                raise ValueError("Cycle detected - no ready nodes")

            layer = [node_map[name] for name in ready]
            layers.append(layer)

            # Remove ready nodes and update in_degrees
            for name in ready:
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
        current_ctx = ctx
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

            # Filter nodes that should run
            nodes_to_run = []
            for node in layer:
                if node.skip_if and node.skip_if(current_ctx):
                    Trace.event("dag_step_skipped", {"step": node.name, "reason": "skip_if"})
                    continue
                nodes_to_run.append(node)

            if not nodes_to_run:
                continue

            # Execute layer in parallel (with semaphore limit)
            semaphore = asyncio.Semaphore(self.max_parallel)

            async def run_node(node: StepNode) -> tuple[str, PipelineContext | Exception]:
                async with semaphore:
                    try:
                        if trace:
                            Trace.event("dag_step_start", {"step": node.name})
                        result = await node.step.run(current_ctx)
                        if trace:
                            Trace.event("dag_step_end", {"step": node.name})
                        return node.name, result
                    except Exception as e:
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
            for name, result in results:
                if isinstance(result, Exception):
                    raise result
                completed_results[name] = result

            # Merge contexts from this layer
            # Properly merge extras from all parallel step results
            merged_extras = dict(current_ctx.extras) if current_ctx.extras else {}
            merged_claims = current_ctx.claims
            merged_sources = current_ctx.sources
            merged_verdict = current_ctx.verdict

            for name, result in results:
                if not isinstance(result, Exception):
                    # Merge extras additively
                    if result.extras:
                        merged_extras.update(result.extras)
                    # Use latest non-empty values
                    if result.claims:
                        merged_claims = result.claims
                    if result.sources:
                        merged_sources = result.sources
                    if result.verdict:
                        merged_verdict = result.verdict

            current_ctx = PipelineContext(
                mode=current_ctx.mode,
                claims=merged_claims,
                lang=current_ctx.lang,
                search_type=current_ctx.search_type,
                gpt_model=current_ctx.gpt_model,
                trace=current_ctx.trace,
                sources=merged_sources,
                evidence=current_ctx.evidence,
                verdict=merged_verdict,
                extras=merged_extras,
            )

        if trace:
            Trace.event(
                "dag_pipeline_end",
                {"mode": self.mode.name, "completed_steps": list(completed_results.keys())},
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
    from spectrue_core.pipeline.steps.decomposed import (
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
            depends_on=["metering_setup"],
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
