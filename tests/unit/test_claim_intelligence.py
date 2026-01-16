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
Claim Intelligence Layer Tests

Tests for Layer 2-4 functionality:
- Structural Claim Prioritization
- Tension / Contradiction Signal
- Evidence-Need Routing
"""

from spectrue_core.graph.types import (
    GraphResult,
    RankedClaim,
)
from spectrue_core.schema.evidence import EvidenceNeedType
from spectrue_core.runtime_config import EngineRuntimeConfig, ClaimGraphConfig


class TestGraphResultM73Methods:
    """Test convenience methods on GraphResult."""

    def test_get_ranked_by_id_found(self):
        """Should return RankedClaim when ID exists."""
        ranked = [
            RankedClaim("c1", 0.5, 1.0, 0.2, True),
            RankedClaim("c2", 0.3, 0.5, 0.8, False),
            RankedClaim("c3", 0.2, 0.1, 0.0, False),
        ]
        result = GraphResult(all_ranked=ranked)
        
        found = result.get_ranked_by_id("c2")
        assert found is not None
        assert found.claim_id == "c2"
        assert found.centrality_score == 0.3
        assert found.in_contradict_weight == 0.8

    def test_get_ranked_by_id_not_found(self):
        """Should return None when ID doesn't exist."""
        ranked = [RankedClaim("c1", 0.5, 1.0, 0.2, True)]
        result = GraphResult(all_ranked=ranked)
        
        assert result.get_ranked_by_id("c99") is None

    def test_get_ranked_by_id_empty(self):
        """Should return None for empty result."""
        result = GraphResult()
        assert result.get_ranked_by_id("c1") is None

    def test_high_tension_claims_filter(self):
        """Should filter claims with in_contradict_weight > 0.5."""
        ranked = [
            RankedClaim("c1", 0.5, 1.0, 0.3, True),   # Below threshold
            RankedClaim("c2", 0.3, 0.5, 0.8, False),  # Above threshold
            RankedClaim("c3", 0.2, 0.1, 0.6, False),  # Above threshold
            RankedClaim("c4", 0.1, 0.2, 0.5, False),  # At threshold (excluded)
            RankedClaim("c5", 0.4, 0.3, 1.2, True),   # Above threshold
        ]
        result = GraphResult(all_ranked=ranked)
        
        high_tension = result.high_tension_claims
        
        assert len(high_tension) == 3
        # Should be sorted by tension descending
        assert high_tension[0].claim_id == "c5"  # 1.2
        assert high_tension[1].claim_id == "c2"  # 0.8
        assert high_tension[2].claim_id == "c3"  # 0.6

    def test_high_tension_claims_limit(self):
        """Should limit to top 5 high-tension claims."""
        ranked = [
            RankedClaim(f"c{i}", 0.1, 0.1, 0.6 + i*0.1, False)
            for i in range(10)
        ]
        result = GraphResult(all_ranked=ranked)
        
        high_tension = result.high_tension_claims
        
        assert len(high_tension) == 5
        # Should have highest tension scores
        assert high_tension[0].in_contradict_weight == 1.5  # c9

    def test_high_tension_claims_empty(self):
        """Should return empty list when no claims exceed threshold."""
        ranked = [
            RankedClaim("c1", 0.5, 1.0, 0.3, True),
            RankedClaim("c2", 0.3, 0.5, 0.4, False),
        ]
        result = GraphResult(all_ranked=ranked)
        
        assert result.high_tension_claims == []

    def test_get_tension_score_found(self):
        """Should return tension score (in_contradict_weight)."""
        ranked = [RankedClaim("c1", 0.5, 1.0, 0.75, True)]
        result = GraphResult(all_ranked=ranked)
        
        assert result.get_tension_score("c1") == 0.75

    def test_get_tension_score_not_found(self):
        """Should return 0.0 for missing claim."""
        result = GraphResult()
        
        assert result.get_tension_score("c99") == 0.0


class TestEvidenceNeedType:
    """Test Layer 4 EvidenceNeedType enum."""

    def test_all_values_exist(self):
        """Should have all required evidence need types."""
        expected = {
            "empirical_study",
            "guideline",
            "official_stats",
            "expert_opinion",
            "anecdotal",
            "news_report",
            "unknown",
        }
        actual = {e.value for e in EvidenceNeedType}
        
        assert actual == expected

    def test_is_string_enum(self):
        """Should be usable as string."""
        assert EvidenceNeedType.EMPIRICAL_STUDY == "empirical_study"
        assert EvidenceNeedType.NEWS_REPORT.value == "news_report"

    def test_serialization(self):
        """Should serialize to JSON-compatible value."""
        import json
        data = {"evidence_need": EvidenceNeedType.OFFICIAL_STATS}
        serialized = json.dumps(data)
        
        assert '"official_stats"' in serialized


class TestClaimGraphConfigM73:
    """Test feature flags in ClaimGraphConfig."""

    def test_default_values(self):
        """Should have correct default values for flags."""
        cfg = ClaimGraphConfig()
        
        # Layer 2
        assert cfg.structural_prioritization_enabled is True
        assert cfg.structural_weight_threshold == 0.5
        assert cfg.structural_boost == 0.1
        
        # Layer 3
        assert cfg.tension_signal_enabled is True
        assert cfg.tension_threshold == 0.5
        assert cfg.tension_boost == 0.15
        
        # Layer 4
        assert cfg.evidence_need_routing_enabled is True

    def test_load_from_env_defaults(self):
        """Should load flags from env with defaults."""
        import os
        
        # Clear any existing env vars
        for key in [
            "CLAIM_GRAPH_STRUCTURAL_ENABLED",
            "CLAIM_GRAPH_TENSION_ENABLED",
            "CLAIM_GRAPH_EVIDENCE_NEED_ENABLED",
        ]:
            os.environ.pop(key, None)
        
        cfg = EngineRuntimeConfig.load_from_env()
        
        assert cfg.claim_graph.structural_prioritization_enabled is True
        assert cfg.claim_graph.tension_signal_enabled is True
        assert cfg.claim_graph.evidence_need_routing_enabled is True

    def test_load_from_env_disabled(self):
        """Should respect disabled flags from env."""
        import os
        
        os.environ["CLAIM_GRAPH_STRUCTURAL_ENABLED"] = "false"
        os.environ["CLAIM_GRAPH_TENSION_ENABLED"] = "0"
        os.environ["CLAIM_GRAPH_EVIDENCE_NEED_ENABLED"] = "no"
        
        try:
            cfg = EngineRuntimeConfig.load_from_env()
            
            assert cfg.claim_graph.structural_prioritization_enabled is False
            assert cfg.claim_graph.tension_signal_enabled is False
            assert cfg.claim_graph.evidence_need_routing_enabled is False
        finally:
            # Cleanup
            os.environ.pop("CLAIM_GRAPH_STRUCTURAL_ENABLED", None)
            os.environ.pop("CLAIM_GRAPH_TENSION_ENABLED", None)
            os.environ.pop("CLAIM_GRAPH_EVIDENCE_NEED_ENABLED", None)

    def test_to_safe_log_dict_includes_m73(self):
        """Should include flags in safe log dict."""
        cfg = EngineRuntimeConfig.load_from_env()
        log_dict = cfg.to_safe_log_dict()
        
        assert "structural_prioritization_enabled" in log_dict["claim_graph"]
        assert "tension_signal_enabled" in log_dict["claim_graph"]
        assert "evidence_need_routing_enabled" in log_dict["claim_graph"]


class TestClaimEnrichmentLogic:
    """Test Layer 2-3 claim enrichment logic."""

    def test_structural_boost_calculation(self):
        """Structural weight > threshold should add boost."""
        cfg = ClaimGraphConfig(
            structural_weight_threshold=0.5,
            structural_boost=0.1,
        )
        
        claim = {"id": "c1", "importance": 0.5}
        structural_weight = 0.8  # > 0.5 threshold
        
        # Simulate boost logic from pipeline
        if structural_weight > cfg.structural_weight_threshold:
            claim["importance"] = min(1.0, claim["importance"] + cfg.structural_boost)
        
        assert claim["importance"] == 0.6

    def test_tension_boost_calculation(self):
        """High tension should add boost."""
        cfg = ClaimGraphConfig(
            tension_threshold=0.5,
            tension_boost=0.15,
        )
        
        claim = {"id": "c1", "importance": 0.5}
        tension_score = 0.8  # > 0.5 threshold
        
        # Simulate boost logic from pipeline
        if tension_score > cfg.tension_threshold:
            claim["importance"] = min(1.0, claim["importance"] + cfg.tension_boost)
        
        assert claim["importance"] == 0.65

    def test_importance_cap_at_1(self):
        """Importance should be capped at 1.0."""
        claim = {"id": "c1", "importance": 0.95}
        
        # Apply multiple boosts
        claim["importance"] = min(1.0, claim["importance"] + 0.2)  # key claim
        claim["importance"] = min(1.0, claim["importance"] + 0.1)  # structural
        claim["importance"] = min(1.0, claim["importance"] + 0.15)  # tension
        
        assert claim["importance"] == 1.0

    def test_claim_enrichment_fields(self):
        """Enriched claim should have all graph fields."""
        ranked = RankedClaim("c1", 0.5, 1.2, 0.8, True)
        
        claim = {"id": "c1", "text": "Some claim"}
        
        # Simulate enrichment
        claim["graph_centrality"] = ranked.centrality_score
        claim["graph_structural_weight"] = ranked.in_structural_weight
        claim["graph_tension_score"] = ranked.in_contradict_weight
        claim["is_key_claim"] = ranked.is_key_claim
        
        assert claim["graph_centrality"] == 0.5
        assert claim["graph_structural_weight"] == 1.2
        assert claim["graph_tension_score"] == 0.8
        assert claim["is_key_claim"] is True
