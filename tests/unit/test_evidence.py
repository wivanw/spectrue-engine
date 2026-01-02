# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


import pytest
from spectrue_core.verification.evidence.evidence import build_evidence_pack
from spectrue_core.verification.evidence.evidence_pack import Claim

@pytest.mark.unit
class TestEvidenceBuilder:
    
    def test_build_minimal_pack(self):
        """Test building pack with minimal inputs."""
        pack = build_evidence_pack(
            fact="Sky is blue",
            claims=None,
            sources=[]
        )
        assert isinstance(pack, dict)
        assert pack["original_fact"] == "Sky is blue"
        # Check generated core claim
        assert len(pack["claims"]) == 1
        assert pack["claims"][0]["type"] == "core"
        # Check constraints default
        assert pack["constraints"]["global_cap"] == 1.0

    def test_domain_deduplication(self):
        """Test sources from same domain are marked is_duplicate."""
        sources = [
            {"title": "S1", "url": "https://bbc.com/news/1"},
            {"title": "S2", "url": "https://bbc.com/sport/2"},
            {"title": "S3", "url": "https://cnn.com/1"},
        ]
        pack = build_evidence_pack(fact="X", claims=None, sources=sources)
        results = pack["search_results"]
        
        # Sort by url or title to be deterministic if needed, 
        # but build_evidence_pack preserves order.
        assert len(results) == 3
        
        # First BBC source is NOT duplicate, second IS
        bbc_results = [r for r in results if "bbc.com" in r["domain"]]
        assert len(bbc_results) == 2
        
        # Logic in code: "is_dup = domain in seen_domains"
        # First one adds to seen, second one sees it.
        assert bbc_results[0]["is_duplicate"] is False
        assert bbc_results[1]["is_duplicate"] is True
        
        # CNN is unique
        cnn_result = [r for r in results if "cnn.com" in r["domain"]][0]
        assert cnn_result["is_duplicate"] is False
        
        # Metrics check
        metrics = pack["metrics"]
        assert metrics["unique_domains"] == 2
        assert metrics["duplicate_ratio"] == 1/3

    def test_source_type_mapping(self):
        """Test evidence_tier -> source_type mapping."""
        sources = [
            {"url": "http://a.gov", "evidence_tier": "A'", "is_trusted": True},
            {"url": "http://b.com", "evidence_tier": "B", "is_trusted": True}, # Trusted => Independent
            {"url": "http://c.com", "evidence_tier": "D", "is_trusted": False},
            {"url": "http://d.com", "evidence_tier": "C", "is_trusted": False},
        ]
        pack = build_evidence_pack(fact="X", claims=None, sources=sources)
        results = pack["search_results"]
        
        types = {r["url"]: r["source_type"] for r in results}
        
        assert types["http://a.gov"] == "official"
        assert types["http://b.com"] == "independent_media"
        assert types["http://c.com"] == "social"
        assert types["http://d.com"] == "aggregator" # Default for unknown/C

    def test_claim_coverage_metrics(self):
        """Test per-claim metrics calculation."""
        claims = [Claim(id="c1", text="C1", search_queries=[])]
        sources = [
            {"claim_id": "c1", "url": "http://x.com", "relevance_score": 0.9},
            {"claim_id": "c1", "url": "http://y.com", "relevance_score": 0.1},
        ]
        pack = build_evidence_pack(fact="X", claims=claims, sources=sources)
        # We need to manually inject claim_id into results if build_evidence_pack doesn't do it from sources dict.
        # WAIT: build_evidence_pack sets claim_id="c1" hardcoded if sources list is passed raw?
        # Let's check code (Line 93): claim_id="c1"
        
        # If we pass `claims` list, build_evidence_pack calculates metrics for them.
        # But if we pass `sources` list (raw dicts), `build_evidence_pack` assigns them all to "c1" (Line 93).
        # So this test only works for single claim unless we use `search_results_clustered`.
        
        metrics = pack["metrics"]["per_claim"]["c1"]
        assert metrics["coverage"] == 0.5 # 1 relevant out of 2
        assert metrics["independent_domains"] == 2

    def test_clustered_evidence_tier_fix(self):
        """Test fix for clustered evidence defaulting to Tier C when trusted."""
        # sources=... is empty, we act as if clustering ran
        clustered_results = [
            {
                "url": "https://phys.org/news/123",
                "domain": "phys.org",
                "title": "Test",
                "snippet": "Test snippet",
                "source_type": "unknown",  # This triggers the default fallback logic we want to bypass if is_trusted
                "is_trusted": True,
                "evidence_tier": None,
                "stance": "support",
                "quote_span": "quote",
                "relevance_score": 1.0, 
                "timeliness_status": "in_window",
                "content_status": "available",
                "claim_id": "c1",
            }
        ]
        
        pack = build_evidence_pack(
            fact="The universe is old.",
            claims=None,
            sources=[], 
            search_results_clustered=clustered_results,
        )
        
        item = pack["items"][0]
        
        # Verification
        # is_trusted=True -> independent_media -> Tier B
        # Before fix: source_type="unknown" -> Tier C
        assert item["tier"] == "B", f"Expected Tier B, got {item['tier']}"
        assert item["channel"] == "reputable_news", f"Expected reputable_news, got {item['channel']}"
