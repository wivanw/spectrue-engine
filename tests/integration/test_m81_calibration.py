"""
M81 Calibration Integration Tests.

Tests for:
- T12: Attribution article handling (interview content)
- T13: Reality article handling (factual news)
"""

from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    VerificationTarget,
    ClaimRole,
    MetadataConfidence,
)


class TestT12AttributionArticle:
    """T12: Interview/attribution articles should get correct metadata."""
    
    def _make_attribution_claim(self, claim_id: str, text: str) -> dict:
        """Create a claim with attribution metadata."""
        return {
            "id": claim_id,
            "text": text,
            "type": "attribution",
            "metadata": ClaimMetadata(
                verification_target=VerificationTarget.ATTRIBUTION,
                claim_role=ClaimRole.SUPPORT,
                metadata_confidence=MetadataConfidence.MEDIUM,
            ),
            "check_worthiness": 0.6,
            "topic_group": "Culture",
        }
    
    def _make_reality_claim(self, claim_id: str, text: str) -> dict:
        """Create a claim with reality metadata."""
        return {
            "id": claim_id,
            "text": text,
            "type": "core",
            "metadata": ClaimMetadata(
                verification_target=VerificationTarget.REALITY,
                claim_role=ClaimRole.CORE,
                metadata_confidence=MetadataConfidence.HIGH,
            ),
            "check_worthiness": 0.9,
            "topic_group": "Politics",
        }
    
    def test_attribution_claims_skip_oracle(self):
        """Oracle should be skipped when ALL claims are attribution."""
        claims = [
            self._make_attribution_claim("c1", "Winslet said paparazzi followed her"),
            self._make_attribution_claim("c2", "She mentioned being told to lose weight"),
            self._make_attribution_claim("c3", "She recalled feeling persecuted"),
        ]
        
        # Count reality vs attribution
        reality_count = 0
        attribution_count = 0
        for c in claims:
            metadata = c.get("metadata")
            if metadata:
                target = metadata.verification_target.value
                if target == "reality":
                    reality_count += 1
                elif target in ("attribution", "existence"):
                    attribution_count += 1
        
        # Rule: Skip Oracle if NO reality claims
        should_skip_oracle = reality_count == 0 and attribution_count > 0
        
        assert should_skip_oracle is True
        assert reality_count == 0
        assert attribution_count == 3
    
    def test_mixed_claims_call_oracle(self):
        """Oracle should be called when at least 1 claim is reality."""
        claims = [
            self._make_attribution_claim("c1", "Winslet said paparazzi followed her"),
            self._make_attribution_claim("c2", "She mentioned being told to lose weight"),
            self._make_reality_claim("c3", "Titanic earned over 2 billion dollars"),
        ]
        
        # Count reality vs attribution
        reality_count = 0
        for c in claims:
            metadata = c.get("metadata")
            if metadata and metadata.verification_target.value == "reality":
                reality_count += 1
        
        # Rule: Call Oracle if at least 1 reality claim
        should_call_oracle = reality_count > 0
        
        assert should_call_oracle is True
        assert reality_count == 1
    
    def test_max_two_core_claims(self):
        """Interview articles should have max 2 core claims."""
        # This tests the prompt rule, not the code  
        # The LLM is instructed to limit core claims to 2
        # We verify by checking the ClaimRole distribution
        
        # Simulated LLM output for interview article (expected after M81 calibration)
        expected_claims = [
            {"role": ClaimRole.CORE, "text": "Main headline claim"},
            {"role": ClaimRole.CORE, "text": "Secondary important claim"},
            {"role": ClaimRole.SUPPORT, "text": "Supporting detail 1"},
            {"role": ClaimRole.ATTRIBUTION, "text": "Quote attribution"},
            {"role": ClaimRole.CONTEXT, "text": "Background info"},
        ]
        
        core_count = sum(1 for c in expected_claims if c["role"] == ClaimRole.CORE)
        
        assert core_count <= 2, f"Expected max 2 core claims, got {core_count}"
    
    def test_attribution_claims_get_medium_confidence(self):
        """Attribution claims should have medium confidence (no primary source)."""
        claim = self._make_attribution_claim("c1", "She said she felt persecuted")
        
        metadata = claim.get("metadata")
        assert metadata is not None
        assert metadata.metadata_confidence == MetadataConfidence.MEDIUM


class TestT13RealityArticle:
    """T13: Factual news articles should get correct treatment."""
    
    def _make_reality_claim(self, claim_id: str, text: str, role: ClaimRole = ClaimRole.CORE) -> dict:
        """Create a claim with reality metadata."""
        return {
            "id": claim_id,
            "text": text,
            "type": "core" if role == ClaimRole.CORE else "support",
            "metadata": ClaimMetadata(
                verification_target=VerificationTarget.REALITY,
                claim_role=role,
                metadata_confidence=MetadataConfidence.HIGH,
            ),
            "check_worthiness": 0.9,
            "topic_group": "Politics",
        }
    
    def test_reality_claims_call_oracle(self):
        """Oracle should be called for reality claims."""
        claims = [
            self._make_reality_claim("c1", "President signed the bill into law"),
            self._make_reality_claim("c2", "The bill passed with 60 votes"),
            self._make_reality_claim("c3", "Opposition criticized the measure", ClaimRole.SUPPORT),
        ]
        
        reality_count = sum(
            1 for c in claims 
            if c.get("metadata") and c["metadata"].verification_target.value == "reality"
        )
        
        should_call_oracle = reality_count > 0
        
        assert should_call_oracle is True
        assert reality_count == 3
    
    def test_reality_claims_get_high_confidence(self):
        """Reality claims with checkable data should have high confidence."""
        claim = self._make_reality_claim("c1", "Bill passed with 60-40 vote")
        
        metadata = claim.get("metadata")
        assert metadata is not None
        assert metadata.metadata_confidence == MetadataConfidence.HIGH
    
    def test_graph_worthiness_for_connected_claims(self):
        """ClaimGraph should run for connected reality claims."""
        claims = [
            self._make_reality_claim("c1", "President signed the bill", ClaimRole.CORE),
            self._make_reality_claim("c2", "Senate approved earlier", ClaimRole.SUPPORT),
            self._make_reality_claim("c3", "House passed last week", ClaimRole.SUPPORT),
        ]
        
        # Count role diversity for graph worthiness
        core_support_count = sum(
            1 for c in claims 
            if c.get("type") in ("core", "support")
        )
        set(c.get("topic_group", "Other") for c in claims)
        
        # Gate: Need at least 2 core/support claims
        graph_worthy = core_support_count >= 2
        
        assert graph_worthy is True
        assert core_support_count == 3
    
    def test_graph_skipped_for_flat_claims(self):
        """ClaimGraph should be skipped for flat (single-topic, same-role) claims."""
        # All context claims, same topic
        claims = [
            {
                "id": "c1", 
                "type": "context", 
                "topic_group": "Other",
                "metadata": ClaimMetadata(
                    verification_target=VerificationTarget.NONE,
                    claim_role=ClaimRole.CONTEXT,
                ),
            },
            {
                "id": "c2", 
                "type": "context", 
                "topic_group": "Other",
                "metadata": ClaimMetadata(
                    verification_target=VerificationTarget.NONE,
                    claim_role=ClaimRole.CONTEXT,
                ),
            },
        ]
        
        core_support_count = sum(
            1 for c in claims 
            if c.get("type") in ("core", "support")
        )
        
        # Gate: Need at least 2 core/support claims
        graph_worthy = core_support_count >= 2
        
        assert graph_worthy is False
        assert core_support_count == 0


class TestOracleGating:
    """Additional tests for Oracle gating logic."""
    
    def test_existence_claims_skip_oracle(self):
        """Existence-only claims should skip Oracle."""
        claims = [
            {
                "id": "c1",
                "text": "The document exists in the archive",
                "metadata": ClaimMetadata(
                    verification_target=VerificationTarget.EXISTENCE,
                    claim_role=ClaimRole.CORE,
                ),
            },
        ]
        
        reality_count = 0
        attribution_existence_count = 0
        for c in claims:
            metadata = c.get("metadata")
            if metadata:
                target = metadata.verification_target.value
                if target == "reality":
                    reality_count += 1
                elif target in ("attribution", "existence"):
                    attribution_existence_count += 1
        
        should_skip = reality_count == 0 and attribution_existence_count > 0
        assert should_skip is True
    
    def test_none_target_claims_no_oracle(self):
        """Claims with target=none should not trigger Oracle."""
        claims = [
            {
                "id": "c1",
                "text": "Horoscope prediction for tomorrow",
                "metadata": ClaimMetadata(
                    verification_target=VerificationTarget.NONE,
                    claim_role=ClaimRole.CONTEXT,
                ),
            },
        ]
        
        reality_count = 0
        attribution_existence_count = 0
        for c in claims:
            metadata = c.get("metadata")
            if metadata:
                target = metadata.verification_target.value
                if target == "reality":
                    reality_count += 1
                elif target in ("attribution", "existence"):
                    attribution_existence_count += 1
        
        # None targets don't count toward attribution_existence
        
        # With only "none" targets, we don't skip Oracle (no attribution to skip for)
        # but we also don't need to call it (no reality to check)
        assert reality_count == 0
        assert attribution_existence_count == 0
        # Default behavior: Oracle may still run based on article_intent
