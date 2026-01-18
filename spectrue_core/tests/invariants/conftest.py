import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from spectrue_core.verification.evidence.evidence_pack import (
    EvidencePack, EvidenceItem, EvidenceMetrics, EvidencePackStats
)
from spectrue_core.agents.llm_client import LLMClient

@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    client.call = AsyncMock()
    # Ensure strict mode behavior works with mocks if needed
    client.call_structured = AsyncMock()
    return client

@pytest.fixture
def empty_evidence_pack():
    return EvidencePack(
        article=None,
        original_fact="Test fact",
        claims=[],
        claim_units=[],
        search_results=[],
        scored_sources=[],
        context_sources=[],
        metrics=EvidenceMetrics(
            total_sources=0,
            unique_domains=0,
            duplicate_ratio=0.0,
            per_claim={},
            overall_coverage=0.0,
            freshness_days_median=None,
            source_type_distribution={},
            per_assertion={}
        ),
        constraints=None,
        claim_id="c1",
        items=[],
        stats=EvidencePackStats(
            domain_diversity=0,
            tiers_present={},
            support_count=0,
            refute_count=0,
            context_count=0,
            outdated_ratio=0.0
        ),
        global_cap=1.0,
        cap_reasons=[]
    )

@pytest.fixture
def sample_claim_result():
    return {
        "status": "ok",
        "rgba": [0.1, 0.8, 0.7, 0.6],
        "explanation": "Test explanation",
        "sources_used": []
    }
