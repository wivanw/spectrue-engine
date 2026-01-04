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
Unit tests for DAG Pipeline execution.
"""

import pytest
from spectrue_core.pipeline.dag import StepNode, DAGPipeline
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import NORMAL_MODE
from spectrue_core.pipeline.errors import PipelineExecutionError


# ─────────────────────────────────────────────────────────────────────────────
# Mock Steps for Testing
# ─────────────────────────────────────────────────────────────────────────────


class MockStep:
    def __init__(self, name: str, result_key: str = None):
        self.name = name
        self.result_key = result_key or name

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        return ctx.set_extra(self.result_key, f"{self.name}_completed")


class FailingStep:
    def __init__(self, name: str):
        self.name = name

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        raise PipelineExecutionError(self.name, "Intentional failure")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: StepNode
# ─────────────────────────────────────────────────────────────────────────────


class TestStepNode:
    def test_step_node_name_from_step(self):
        step = MockStep("test_step")
        node = StepNode(step=step)
        assert node.name == "test_step"

    def test_step_node_with_dependencies(self):
        step = MockStep("child")
        node = StepNode(step=step, depends_on=["parent1", "parent2"])
        assert node.depends_on == ["parent1", "parent2"]

    def test_step_node_optional_flag(self):
        step = MockStep("optional_step")
        node = StepNode(step=step, optional=True)
        assert node.optional is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DAGPipeline Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestDAGPipelineValidation:
    def test_dag_rejects_unknown_dependency(self):
        nodes = [
            StepNode(step=MockStep("step1"), depends_on=["nonexistent"]),
        ]
        with pytest.raises(ValueError, match="unknown step"):
            DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)

    def test_dag_detects_simple_cycle(self):
        nodes = [
            StepNode(step=MockStep("step1"), depends_on=["step2"]),
            StepNode(step=MockStep("step2"), depends_on=["step1"]),
        ]
        with pytest.raises(ValueError, match="Cycle"):
            DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)

    def test_dag_accepts_valid_dependencies(self):
        nodes = [
            StepNode(step=MockStep("step1")),
            StepNode(step=MockStep("step2"), depends_on=["step1"]),
            StepNode(step=MockStep("step3"), depends_on=["step1", "step2"]),
        ]
        dag = DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)
        assert len(dag.nodes) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DAGPipeline Execution
# ─────────────────────────────────────────────────────────────────────────────


class TestDAGPipelineExecution:
    @pytest.mark.asyncio
    async def test_execute_single_step(self):
        nodes = [StepNode(step=MockStep("only_step"))]
        dag = DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)

        ctx = PipelineContext(mode=NORMAL_MODE)
        result = await dag.run(ctx)

        assert result.get_extra("only_step") == "only_step_completed"

    @pytest.mark.asyncio
    async def test_execute_linear_chain(self):
        nodes = [
            StepNode(step=MockStep("step1")),
            StepNode(step=MockStep("step2"), depends_on=["step1"]),
            StepNode(step=MockStep("step3"), depends_on=["step2"]),
        ]
        dag = DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)

        ctx = PipelineContext(mode=NORMAL_MODE)
        result = await dag.run(ctx)

        assert result.get_extra("step1") == "step1_completed"
        assert result.get_extra("step2") == "step2_completed"
        assert result.get_extra("step3") == "step3_completed"

    @pytest.mark.asyncio
    async def test_execute_parallel_layer(self):
        """Steps without dependencies should run in parallel."""
        nodes = [
            StepNode(step=MockStep("parallel1")),
            StepNode(step=MockStep("parallel2")),
            StepNode(step=MockStep("parallel3")),
        ]
        dag = DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)

        ctx = PipelineContext(mode=NORMAL_MODE)
        result = await dag.run(ctx)

        # All completed (order may vary)
        assert result.get_extra("parallel1") == "parallel1_completed"
        assert result.get_extra("parallel2") == "parallel2_completed"
        assert result.get_extra("parallel3") == "parallel3_completed"

    @pytest.mark.asyncio
    async def test_optional_step_failure_continues(self):
        """Optional step failure should not stop pipeline."""
        nodes = [
            StepNode(step=MockStep("before")),
            StepNode(step=FailingStep("failing"), depends_on=["before"], optional=True),
            StepNode(step=MockStep("after"), depends_on=["failing"]),
        ]
        dag = DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)

        ctx = PipelineContext(mode=NORMAL_MODE)
        result = await dag.run(ctx)

        assert result.get_extra("before") == "before_completed"
        assert result.get_extra("after") == "after_completed"

    @pytest.mark.asyncio
    async def test_required_step_failure_stops(self):
        """Required step failure should stop pipeline."""
        nodes = [
            StepNode(step=MockStep("before")),
            StepNode(step=FailingStep("failing"), depends_on=["before"]),
            StepNode(step=MockStep("after"), depends_on=["failing"]),
        ]
        dag = DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)

        ctx = PipelineContext(mode=NORMAL_MODE)
        with pytest.raises(PipelineExecutionError):
            await dag.run(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Topological Sort
# ─────────────────────────────────────────────────────────────────────────────


class TestTopologicalSort:
    def test_sort_linear_chain(self):
        nodes = [
            StepNode(step=MockStep("c"), depends_on=["b"]),
            StepNode(step=MockStep("a")),
            StepNode(step=MockStep("b"), depends_on=["a"]),
        ]
        dag = DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)
        layers = dag._topological_sort()

        # Layer 0: a (no deps)
        # Layer 1: b (depends on a)
        # Layer 2: c (depends on b)
        assert len(layers) == 3
        assert layers[0][0].name == "a"
        assert layers[1][0].name == "b"
        assert layers[2][0].name == "c"

    def test_sort_diamond(self):
        """Diamond pattern: A -> B,C -> D"""
        nodes = [
            StepNode(step=MockStep("a")),
            StepNode(step=MockStep("b"), depends_on=["a"]),
            StepNode(step=MockStep("c"), depends_on=["a"]),
            StepNode(step=MockStep("d"), depends_on=["b", "c"]),
        ]
        dag = DAGPipeline(mode=NORMAL_MODE, nodes=nodes, validate_mode_contract=False)
        layers = dag._topological_sort()

        # Layer 0: a
        # Layer 1: b, c (parallel)
        # Layer 2: d
        assert len(layers) == 3
        layer1_names = {n.name for n in layers[1]}
        assert layer1_names == {"b", "c"}
