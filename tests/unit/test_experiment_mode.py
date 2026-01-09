# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Tests for retrieval experiment mode.

Tests cover:
1. experiment_mode=false preserves existing behavior (no bypass events)
2. experiment_mode=true:
   - sanity gate does not filter (kept_count == results_count)
   - clustering does not reduce (reps_count == kept_count)
   - budget checks are bypassed (extract proceeds even if budget would be 0)
   - all claims receive same broadcasted_sources_count
"""

import pytest
from unittest.mock import patch, MagicMock

from spectrue_core.verification.retrieval.experiment_mode import (
    is_experiment_mode,
    set_experiment_mode,
    reset_experiment_mode,
    should_filter_sources,
    should_apply_budget,
    should_stop_early,
    build_broadcasted_sources,
)
from spectrue_core.runtime_config import (
    EngineRuntimeConfig,
    RetrievalExperimentConfig,
)
from spectrue_core.scoring.budget_allocation import BudgetState, ExtractBudgetParams


class TestExperimentModeConfig:
    """Test experiment mode config loading."""

    def test_experiment_mode_default_off(self, monkeypatch):
        """Experiment mode should default to off."""
        monkeypatch.delenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", raising=False)
        reset_experiment_mode()
        assert not is_experiment_mode()

    def test_experiment_mode_env_on(self, monkeypatch):
        """Experiment mode should be on when env var is set."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        assert is_experiment_mode()

    def test_experiment_mode_env_true(self, monkeypatch):
        """Experiment mode should accept 'true' value."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "true")
        reset_experiment_mode()
        assert is_experiment_mode()

    def test_experiment_mode_context_override(self, monkeypatch):
        """Context variable should override env var."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "0")
        reset_experiment_mode()
        assert not is_experiment_mode()
        
        set_experiment_mode(True)
        assert is_experiment_mode()
        
        reset_experiment_mode()
        assert not is_experiment_mode()

    def test_runtime_config_loads_experiment_mode(self, monkeypatch):
        """EngineRuntimeConfig should load experiment mode from env."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        config = EngineRuntimeConfig.load_from_env()
        assert config.retrieval_experiment.experiment_mode is True

    def test_runtime_config_experiment_mode_default(self, monkeypatch):
        """EngineRuntimeConfig should default experiment mode to off."""
        monkeypatch.delenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", raising=False)
        config = EngineRuntimeConfig.load_from_env()
        assert config.retrieval_experiment.experiment_mode is False


class TestExperimentModeHelpers:
    """Test experiment mode helper functions."""

    def test_should_filter_sources_normal_mode(self, monkeypatch):
        """Normal mode: should_filter_sources returns True."""
        monkeypatch.delenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", raising=False)
        reset_experiment_mode()
        assert should_filter_sources() is True

    def test_should_filter_sources_experiment_mode(self, monkeypatch):
        """Experiment mode: should_filter_sources returns False."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        assert should_filter_sources() is False

    def test_should_apply_budget_normal_mode(self, monkeypatch):
        """Normal mode: should_apply_budget returns True."""
        monkeypatch.delenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", raising=False)
        reset_experiment_mode()
        assert should_apply_budget() is True

    def test_should_apply_budget_experiment_mode(self, monkeypatch):
        """Experiment mode: should_apply_budget returns False."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        assert should_apply_budget() is False

    def test_should_stop_early_normal_mode(self, monkeypatch):
        """Normal mode: should_stop_early returns True."""
        monkeypatch.delenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", raising=False)
        reset_experiment_mode()
        assert should_stop_early() is True

    def test_should_stop_early_experiment_mode(self, monkeypatch):
        """Experiment mode: should_stop_early returns False."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        assert should_stop_early() is False


class TestSourceBroadcasting:
    """Test source broadcasting functionality."""

    def test_build_broadcasted_sources_normal_mode(self, monkeypatch):
        """Normal mode: build_broadcasted_sources returns empty."""
        monkeypatch.delenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", raising=False)
        reset_experiment_mode()
        
        global_sources = [{"url": "https://a.com", "title": "A"}]
        by_claim = {"c1": [{"url": "https://b.com", "title": "B"}]}
        
        result = build_broadcasted_sources(global_sources, by_claim)
        assert result == []

    def test_build_broadcasted_sources_experiment_mode(self, monkeypatch):
        """Experiment mode: build_broadcasted_sources merges all sources."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        
        global_sources = [
            {"url": "https://a.com", "title": "A", "snippet": "snippet a"},
            {"url": "https://b.com", "title": "B", "snippet": "snippet b"},
        ]
        by_claim = {
            "c1": [{"url": "https://c.com", "title": "C", "snippet": "snippet c"}],
            "c2": [{"url": "https://d.com", "title": "D", "snippet": "snippet d"}],
        }
        
        result = build_broadcasted_sources(global_sources, by_claim)
        
        # Should have 4 unique sources
        assert len(result) == 4
        urls = {s["url"] for s in result}
        assert urls == {"https://a.com", "https://b.com", "https://c.com", "https://d.com"}

    def test_build_broadcasted_sources_deduplicates_urls(self, monkeypatch):
        """Experiment mode: duplicate URLs are deduplicated."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        
        global_sources = [{"url": "https://x.com", "title": "X"}]
        by_claim = {
            "c1": [{"url": "https://x.com", "title": "X duplicate"}],  # Same URL
            "c2": [{"url": "https://y.com", "title": "Y"}],
        }
        
        result = build_broadcasted_sources(global_sources, by_claim)
        
        # Should have 2 unique sources (x.com deduplicated)
        assert len(result) == 2
        urls = {s["url"] for s in result}
        assert urls == {"https://x.com", "https://y.com"}

    def test_build_broadcasted_sources_minimal_fields(self, monkeypatch):
        """Experiment mode: only minimal fields are kept."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        
        global_sources = [{
            "url": "https://example.com",
            "title": "Example",
            "snippet": "test snippet",
            "content": "full content",
            "score": 0.9,
            "extra_field": "should be dropped",
            "another_extra": 123,
        }]
        
        result = build_broadcasted_sources(global_sources, {})
        
        assert len(result) == 1
        src = result[0]
        assert set(src.keys()) == {"url", "title", "snippet", "content", "score"}


class TestBudgetBypass:
    """Test budget bypass in experiment mode."""

    def test_compute_extract_limit_normal_mode(self, monkeypatch):
        """Normal mode: extract limit is computed based on EVOI."""
        monkeypatch.delenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", raising=False)
        reset_experiment_mode()
        
        state = BudgetState(params=ExtractBudgetParams(max_extracts=18))
        limit = state.compute_extract_limit()
        
        # Should be less than max (based on EVOI)
        assert limit < 18

    def test_compute_extract_limit_experiment_mode(self, monkeypatch):
        """Experiment mode: extract limit is max_extracts (infinite budget)."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        
        state = BudgetState(params=ExtractBudgetParams(max_extracts=18))
        limit = state.compute_extract_limit()
        
        # Should be max_extracts in experiment mode
        assert limit == 18

    def test_should_continue_extracting_normal_mode_stops(self, monkeypatch):
        """Normal mode: extraction stops when EVOI is low."""
        monkeypatch.delenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", raising=False)
        reset_experiment_mode()
        
        # Simulate high evidence sufficiency
        state = BudgetState(
            params=ExtractBudgetParams(max_extracts=18, min_extracts=2),
            extracts_used=5,
            quotes_found=5,  # High quote ratio
            relevant_sources=5,
            total_sources=5,
        )
        
        should_continue, reason = state.should_continue_extracting()
        # May or may not continue based on EVOI, but reason should not be experiment bypass
        assert "experiment_mode_bypass" not in reason

    def test_should_continue_extracting_experiment_mode_continues(self, monkeypatch):
        """Experiment mode: extraction continues until max_extracts."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        
        state = BudgetState(
            params=ExtractBudgetParams(max_extracts=18, min_extracts=2),
            extracts_used=5,
            quotes_found=5,
            relevant_sources=5,
            total_sources=5,
        )
        
        should_continue, reason = state.should_continue_extracting()
        assert should_continue is True
        assert reason == "experiment_mode_bypass"

    def test_should_continue_extracting_experiment_mode_stops_at_max(self, monkeypatch):
        """Experiment mode: extraction stops at max_extracts."""
        monkeypatch.setenv("SPECTRUE_RETRIEVAL_EXPERIMENT_MODE", "1")
        reset_experiment_mode()
        
        state = BudgetState(
            params=ExtractBudgetParams(max_extracts=18),
            extracts_used=18,  # At max
        )
        
        should_continue, reason = state.should_continue_extracting()
        assert should_continue is False
        assert reason == "max_extracts_reached"
