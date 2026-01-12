# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Unit tests for BuildQueriesStep fallback behavior (M130).

Tests cover:
- Fallback when doc-plan returns empty
- Terminal empty error handling
- Trace event verification
"""

import pytest
from unittest.mock import MagicMock, patch
from spectrue_core.pipeline.steps.retrieval.build_queries import BuildQueriesStep
from spectrue_core.pipeline.core import PipelineContext


@pytest.fixture
def mock_pipeline_context():
    """Create a mock PipelineContext with claims."""
    ctx = MagicMock(spec=PipelineContext)
    ctx.claims = [
        {
            "id": "c1",
            "text": "Test claim",
            "context_entities": ["entity1", "entity2"],
            "subject_entities": ["subject1"],
        }
    ]
    ctx.mode = MagicMock()
    ctx.mode.name = "standard"
    ctx.search_type = "general"
    ctx.get_extra = MagicMock(side_effect=lambda key, default=None: {
        "target_claims": ctx.claims,
        "prepared_fact": "Test fact with entities",
        "input_text": "Test input",
    }.get(key, default))
    ctx.with_update = MagicMock(return_value=ctx)
    ctx.set_extra = MagicMock(return_value=ctx)
    return ctx


@pytest.fixture
def mock_empty_claims_context():
    """Create a mock PipelineContext with claims lacking entities."""
    ctx = MagicMock(spec=PipelineContext)
    ctx.claims = [
        {
            "id": "c1",
            "text": "Test claim without entities",
        }
    ]
    ctx.mode = MagicMock()
    ctx.mode.name = "standard"
    ctx.search_type = "general"
    ctx.get_extra = MagicMock(side_effect=lambda key, default=None: {
        "target_claims": ctx.claims,
        "prepared_fact": "",
        "input_text": "",
    }.get(key, default))
    ctx.with_update = MagicMock(return_value=ctx)
    ctx.set_extra = MagicMock(return_value=ctx)
    return ctx


class TestBuildQueriesStepFallback:
    """Tests for BuildQueriesStep fallback behavior (A)."""

    @pytest.mark.asyncio
    async def test_uses_cegs_queries_when_available(self, mock_pipeline_context):
        """When CEGS returns queries, they are used."""
        step = BuildQueriesStep()
        
        with patch("spectrue_core.pipeline.steps.retrieval.build_queries.build_doc_query_plan") as mock_plan:
            mock_plan.return_value = ["query1", "query2"]
            
            await step.run(mock_pipeline_context)
            
            mock_plan.assert_called_once()
            # Should have called set_extra with search_queries
            calls = mock_pipeline_context.set_extra.call_args_list
            search_queries_call = next(
                (c for c in calls if c[0][0] == "search_queries"), 
                None
            )
            assert search_queries_call is not None
            assert search_queries_call[0][1] == ["query1", "query2"]

    @pytest.mark.asyncio
    async def test_fallback_on_empty_doc_plan(self, mock_empty_claims_context):
        """Fallback to legacy when CEGS returns empty."""
        step = BuildQueriesStep()
        
        trace_events = []
        
        with patch("spectrue_core.pipeline.steps.retrieval.build_queries.build_doc_query_plan") as mock_plan:
            mock_plan.return_value = []
            
            with patch("spectrue_core.pipeline.steps.retrieval.build_queries.select_diverse_queries") as mock_legacy:
                mock_legacy.return_value = ["fallback_query"]
                
                with patch("spectrue_core.pipeline.steps.retrieval.build_queries.Trace") as mock_trace:
                    mock_trace.event = MagicMock(side_effect=lambda name, data: trace_events.append((name, data)))
                    
                    await step.run(mock_empty_claims_context)
                    
                    # Should have called legacy fallback
                    mock_legacy.assert_called_once()
                    
                    # Should have emitted fallback trace events
                    event_names = [e[0] for e in trace_events]
                    assert "retrieval.doc_plan.empty" in event_names
                    assert "retrieval.doc_plan.fallback" in event_names

    @pytest.mark.asyncio
    async def test_terminal_empty_raises_error(self, mock_empty_claims_context):
        """Raises ValueError when all fallbacks fail."""
        step = BuildQueriesStep()
        
        with patch("spectrue_core.pipeline.steps.retrieval.build_queries.build_doc_query_plan") as mock_plan:
            mock_plan.return_value = []
            
            with patch("spectrue_core.pipeline.steps.retrieval.build_queries.select_diverse_queries") as mock_legacy:
                # Legacy also returns empty
                mock_legacy.return_value = []
                
                with patch("spectrue_core.pipeline.steps.retrieval.build_queries.Trace"):
                    with pytest.raises(ValueError) as exc_info:
                        await step.run(mock_empty_claims_context)
                    
                    assert "PIPELINE_ERROR" in str(exc_info.value)
                    assert "0 global queries" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self, mock_pipeline_context):
        """Fallback to legacy when CEGS raises exception."""
        step = BuildQueriesStep()
        
        with patch("spectrue_core.pipeline.steps.retrieval.build_queries.build_doc_query_plan") as mock_plan:
            mock_plan.side_effect = RuntimeError("CEGS failed")
            
            with patch("spectrue_core.pipeline.steps.retrieval.build_queries.select_diverse_queries") as mock_legacy:
                mock_legacy.return_value = ["fallback_query"]
                
                with patch("spectrue_core.pipeline.steps.retrieval.build_queries.Trace"):
                    await step.run(mock_pipeline_context)
                    
                    # Should have called legacy fallback
                    mock_legacy.assert_called_once()


class TestTraceEventConsistency:
    """Tests for trace event schema consistency (F)."""

    @pytest.mark.asyncio
    async def test_retrieval_plan_trace_emitted(self, mock_pipeline_context):
        """retrieval.plan trace event is emitted."""
        step = BuildQueriesStep()
        trace_events = []
        
        with patch("spectrue_core.pipeline.steps.retrieval.build_queries.build_doc_query_plan") as mock_plan:
            mock_plan.return_value = ["q1", "q2"]
            
            with patch("spectrue_core.pipeline.steps.retrieval.build_queries.Trace") as mock_trace:
                mock_trace.event = MagicMock(side_effect=lambda name, data: trace_events.append((name, data)))
                
                await step.run(mock_pipeline_context)
                
                event_names = [e[0] for e in trace_events]
                assert "retrieval.plan" in event_names
                
                # Check retrieval.plan has required fields
                plan_event = next(e for e in trace_events if e[0] == "retrieval.plan")
                assert "plan_id" in plan_event[1]
                assert "mode" in plan_event[1]
                assert "global_queries" in plan_event[1]
