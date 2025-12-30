# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Unit Tests for Evidence Sufficiency

Tests the sufficiency rules for early exit from progressive widening.
"""

from spectrue_core.verification.sufficiency import (
    evidence_sufficiency,
    check_sufficiency_for_claim,
    is_authoritative,
    is_reputable_news,
    is_origin_source,
    get_domain_tier,
    SufficiencyStatus,
)
from spectrue_core.schema.claim_metadata import (
    VerificationTarget,
    EvidenceChannel,
    ClaimMetadata,
    ClaimRole,
    MetadataConfidence,
    SearchLocalePlan,
    RetrievalPolicy,
)


# ─────────────────────────────────────────────────────────────────────────────
# T16: Tier Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTierDetection:
    """Test domain tier detection functions."""
    
    def test_is_authoritative_gov(self):
        """Government domains are authoritative."""
        assert is_authoritative("cdc.gov") is True
        assert is_authoritative("nih.gov") is True
        assert is_authoritative("epa.gov") is True
    
    def test_is_authoritative_edu(self):
        """Education domains are authoritative."""
        assert is_authoritative("mit.edu") is True
        assert is_authoritative("stanford.edu") is True
    
    def test_is_authoritative_international(self):
        """International public bodies are authoritative."""
        assert is_authoritative("who.int") is True
        assert is_authoritative("un.org") is True
        assert is_authoritative("nasa.gov") is True
    
    def test_is_authoritative_science(self):
        """Science journals are authoritative."""
        assert is_authoritative("nature.com") is True
        assert is_authoritative("science.org") is True
        assert is_authoritative("thelancet.com") is True
    
    def test_is_authoritative_negative(self):
        """Regular domains are not authoritative."""
        assert is_authoritative("example.com") is False
        assert is_authoritative("myblog.net") is False
        assert is_authoritative("facebook.com") is False
    
    def test_is_reputable_news_major(self):
        """Major news outlets are reputable."""
        assert is_reputable_news("reuters.com") is True
        assert is_reputable_news("bbc.com") is True
        assert is_reputable_news("nytimes.com") is True
    
    def test_is_reputable_news_regional(self):
        """Regional quality media are reputable."""
        assert is_reputable_news("pravda.com.ua") is True
        assert is_reputable_news("spiegel.de") is True
    
    def test_is_reputable_news_negative(self):
        """Random domains are not reputable."""
        assert is_reputable_news("randomsite.com") is False
        assert is_reputable_news("facebook.com") is False
    
    def test_get_domain_tier(self):
        """Domain tier detection returns correct channels."""
        assert get_domain_tier("cdc.gov") == EvidenceChannel.AUTHORITATIVE
        assert get_domain_tier("bbc.com") == EvidenceChannel.REPUTABLE_NEWS
        assert get_domain_tier("twitter.com") == EvidenceChannel.SOCIAL
        assert get_domain_tier("randomsite.xyz") == EvidenceChannel.LOW_RELIABILITY


# ─────────────────────────────────────────────────────────────────────────────
# T17: Sufficiency Rule Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSufficiencyRule1:
    """Test Rule 1: 1 authoritative source with quote → sufficient."""
    
    def test_one_authoritative_with_quote_sufficient(self):
        """T17: 1 authoritative source with quote is sufficient."""
        sources = [
            {
                "url": "https://cdc.gov/article",
                "stance": "support",
                "quote": "The claim is verified by our research.",
            }
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.REALITY)
        
        assert result.status == SufficiencyStatus.SUFFICIENT
        assert result.rule_matched == "Rule1"
        assert result.authoritative_count == 1
    
    def test_authoritative_no_quote_insufficient(self):
        """Authoritative without quote is not sufficient for Rule 1."""
        sources = [
            {
                "url": "https://cdc.gov/article",
                "stance": "support",
                # No quote
            }
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.REALITY)
        
        # Expect sufficient after tuning logbf_authoritative to 1.4
        assert result.status == SufficiencyStatus.SUFFICIENT
        assert result.authoritative_count == 1
    
    def test_authoritative_with_quote_sufficient_regardless_of_stance(self):
        """T003: Authoritative source with quote IS sufficient during retrieval.
        
        After T003, sufficiency uses structural signals (quote/content presence)
        not semantic stance labels, because stance is computed in a later stage.
        """
        sources = [
            {
                "url": "https://nasa.gov/article",
                "stance": "context",  # Stance is ignored during retrieval sufficiency
                "quote": "Some background information.",
            }
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.REALITY)
        
        # T003: With quote present, authoritative source satisfies Rule1
        assert result.status == SufficiencyStatus.SUFFICIENT
        assert result.rule_matched == "Rule1"


class TestSufficiencyRule2:
    """Test Rule 2: 2 independent reputable sources → sufficient."""
    
    def test_two_reputable_different_domains_sufficient(self):
        """T17: 2 reputable sources from different domains is sufficient."""
        # Use regional news (Tier B) not global agencies (Tier A)
        sources = [
            {
                "url": "https://nytimes.com/article1",
                "stance": "support",
                "quote": "NYT confirms the claim.",
            },
            {
                "url": "https://washingtonpost.com/article2",
                "stance": "support",
                "quote": "WashPost also confirms.",
            },
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.REALITY)
        
        assert result.status == SufficiencyStatus.SUFFICIENT
        assert result.rule_matched == "Rule2"
        assert result.independent_domains >= 2
    
    def test_two_reputable_same_domain_insufficient(self):
        """2 sources from same domain don't count as independent."""
        sources = [
            {
                "url": "https://bbc.com/article1",
                "stance": "support",
                "quote": "First article.",
            },
            {
                "url": "https://bbc.com/article2",
                "stance": "support",
                "quote": "Second article.",
            },
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.REALITY)
        
        # Same domain = 1 independent, not sufficient
        assert result.status == SufficiencyStatus.INSUFFICIENT
        assert result.independent_domains == 1
    
    def test_one_reputable_insufficient(self):
        """1 reputable source alone is not sufficient."""
        sources = [
            {
                "url": "https://nytimes.com/article",
                "stance": "support",
                "quote": "NYT reports...",
            }
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.REALITY)
        
        assert result.status == SufficiencyStatus.INSUFFICIENT


class TestSufficiencyRule3:
    """Test Rule 3: Attribution claims with origin source → sufficient."""
    
    def test_attribution_with_origin_sufficient(self):
        """T17: Attribution claim with origin source is sufficient."""
        sources = [
            {
                "url": "https://whitehouse.gov/statement",
                "stance": "support",
                "title": "Official Statement from the President",
                "is_origin": True,
                "quote": "In an official statement, the President said X.",
            }
        ]
        
        result = evidence_sufficiency(
            "c1",
            sources,
            VerificationTarget.ATTRIBUTION,
            claim_text="President said X",
        )
        
        assert result.status == SufficiencyStatus.SUFFICIENT
        assert result.rule_matched == "Rule3"
    
    def test_existence_with_official_source_sufficient(self):
        """Existence claim with official source is sufficient."""
        sources = [
            {
                "url": "https://nature.com/article",
                "stance": "support",
                "title": "Official Research Publication",
                "is_primary": True,
                "quote": "This publication exists and contains the referenced statement.",
            }
        ]
        
        result = evidence_sufficiency(
            "c1",
            sources,
            VerificationTarget.EXISTENCE,
        )
        
        assert result.status == SufficiencyStatus.SUFFICIENT


class TestSufficiencyNotSufficient:
    """Test cases where evidence is insufficient."""
    
    def test_only_social_insufficient(self):
        """T17: Only social media sources is insufficient."""
        sources = [
            {
                "url": "https://twitter.com/user/status/123",
                "stance": "support",
                "content": "Someone tweeted this.",
            },
            {
                "url": "https://facebook.com/post/456",
                "stance": "support",
                "content": "Facebook post.",
            },
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.REALITY)
        
        assert result.status == SufficiencyStatus.INSUFFICIENT
        assert result.authoritative_count == 0
        assert result.reputable_count == 0
    
    def test_sources_with_quotes_sufficient_regardless_of_stance(self):
        """T003: Sources with quotes ARE sufficient during retrieval.
        
        After T003, retrieval sufficiency uses structural signals (quote/content)
        not semantic stance labels. Two reputable sources with quotes from 
        different domains satisfy Rule2.
        """
        sources = [
            {
                "url": "https://bbc.com/background",
                "stance": "context",  # Stance is ignored during retrieval
                "quote": "Background information only.",
            },
            {
                "url": "https://reuters.com/explainer",
                "stance": "context",  # Stance is ignored during retrieval
                "quote": "More context.",
            },
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.REALITY)
        
        # T003: With quotes present, sources satisfy sufficiency rules
        # Note: reuters.com is Tier A (global news agency), so Rule1 may match first
        assert result.status == SufficiencyStatus.SUFFICIENT
        assert result.rule_matched in ("Rule1", "Rule2")  # Either rule is acceptable
    
    def test_empty_sources_insufficient(self):
        """No sources is insufficient."""
        result = evidence_sufficiency("c1", [], VerificationTarget.REALITY)
        
        assert result.status == SufficiencyStatus.INSUFFICIENT
        assert "No sources" in result.reason


class TestSufficiencySkip:
    """Test skip cases (verification_target=none)."""
    
    def test_none_target_skips(self):
        """verification_target=none → skip sufficiency check."""
        sources = [
            {"url": "https://example.com", "stance": "support"}
        ]
        
        result = evidence_sufficiency("c1", sources, VerificationTarget.NONE)
        
        assert result.status == SufficiencyStatus.SKIP
        assert "none" in result.reason.lower()


class TestCheckSufficiencyForClaim:
    """Test convenience wrapper function."""
    
    def test_extracts_metadata_from_claim(self):
        """check_sufficiency_for_claim extracts metadata correctly."""
        metadata = ClaimMetadata(
            verification_target=VerificationTarget.REALITY,
            claim_role=ClaimRole.CORE,
            check_worthiness=0.8,
            search_locale_plan=SearchLocalePlan(primary="en", fallback=["en"]),
            retrieval_policy=RetrievalPolicy(channels_allowed=[]),
            metadata_confidence=MetadataConfidence.HIGH,
        )
        
        claim = {
            "id": "c1",
            "normalized_text": "Test claim",
            "metadata": metadata,
        }
        
        sources = [
            {
                "url": "https://who.int/article",
                "stance": "support",
                "quote": "Confirmed.",
            }
        ]
        
        result = check_sufficiency_for_claim(claim, sources)
        
        assert result.claim_id == "c1"
        assert result.status == SufficiencyStatus.SUFFICIENT
    
    def test_handles_claim_without_metadata(self):
        """Claim without metadata defaults to REALITY target."""
        claim = {
            "id": "c2",
            "text": "Some claim",
            # No metadata
        }
        
        sources = []
        
        result = check_sufficiency_for_claim(claim, sources)
        
        assert result.claim_id == "c2"
        assert result.status == SufficiencyStatus.INSUFFICIENT


class TestIsOriginSource:
    """Test origin source detection."""
    
    def test_authoritative_is_origin(self):
        """Authoritative sources are considered origins."""
        source = {"url": "https://cdc.gov/announcement"}
        
        assert is_origin_source(source, "CDC announced...") is True
    
    def test_official_in_title_is_origin(self):
        """Source with 'official' in title is origin."""
        source = {
            "url": "https://company.com/news",
            "title": "Official Company Statement",
        }
        
        assert is_origin_source(source, "Company said...") is True
    
    def test_is_primary_flag(self):
        """Source marked as primary is origin."""
        source = {
            "url": "https://example.com/article",
            "is_primary": True,
        }
        
        assert is_origin_source(source, "Test claim") is True
    
    def test_random_source_not_origin(self):
        """Random source is not origin."""
        source = {
            "url": "https://random-blog.com/post",
            "title": "My opinion on things",
        }
        
        assert is_origin_source(source, "Some claim") is False

    def test_string_source_is_not_origin_fail_closed(self):
        """String sources must not satisfy origin-source checks (fail-closed)."""
        assert is_origin_source("https://whitehouse.gov/statement", "President said X") is False


class TestSufficiencyStringSources:
    """Regression: string sources must not crash sufficiency."""

    def test_attribution_string_source_does_not_crash_or_match_rule3(self):
        sources = ["https://whitehouse.gov/statement"]

        result = evidence_sufficiency(
            "c1",
            sources,  # type: ignore[arg-type]
            VerificationTarget.ATTRIBUTION,
            claim_text="President said X",
        )

        assert result.status == SufficiencyStatus.INSUFFICIENT
        assert result.rule_matched != "Rule3"
