import pytest
import asyncio
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import get_mode
from spectrue_core.pipeline.contracts import EVIDENCE_INDEX_KEY, EvidenceIndex, EvidencePackContract, EvidenceItem
from spectrue_core.pipeline.steps.evidence_spillover import EvidenceSpilloverStep

@dataclass
class MockConfig:
    runtime: Any = None

@dataclass
class MockRuntime:
    deep_v2: Any = None

@dataclass
class MockDeepV2Config:
    corroboration_top_k: int = 3

def test_evidence_spillover_general_mode():
    # 1. Setup context
    mode = get_mode("general")
    claims = [
        {"id": "c1", "text": "Claim 1", "assertions": [{"key": "a1", "dimension": "FACT"}]},
        {"id": "c2", "text": "Claim 2", "assertions": [{"key": "a1", "dimension": "FACT"}]},
    ]
    
    # Source for c1
    sources = [{
        "url": "https://example.com/1",
        "claim_id": "c1",
        "stance": "SUPPORT",
        "relevance_score": 0.9,
        "quote": "Quote for c1",
        "assertion_key": "a1"
    }]
    
    # EvidenceIndex for c1
    pack_c1 = EvidencePackContract(
        items=(EvidenceItem(url="https://example.com/1", quote="Quote for c1", stance="SUPPORT"),),
        stats={"n_total": 1}
    )
    index = EvidenceIndex(
        by_claim_id={"c1": pack_c1},
        global_pack=None,
        stats={"claims_total": 2}
    )
    
    # Global EvidencePack (dict for standard mode)
    evidence_pack = {
        "items": [sources[0]],
        "stats": {"n_total": 1}
    }
    
    ctx = (
        PipelineContext(mode=mode)
        .with_update(claims=claims, sources=sources, evidence=evidence_pack)
        .set_extra("cluster_map", {"c1": "cluster_1", "c2": "cluster_1"})
        .set_extra(EVIDENCE_INDEX_KEY, index)
    )
    
    # 2. Setup Step
    config = MockConfig(runtime=MockRuntime(deep_v2=MockDeepV2Config()))
    step = EvidenceSpilloverStep(config=config)
    
    # 3. Run Step
    loop = asyncio.get_event_loop()
    new_ctx = loop.run_until_complete(step.run(ctx))
    
    # 4. Verify c2 has evidence in sources
    c2_sources = [s for s in new_ctx.sources if s.get("claim_id") == "c2"]
    assert len(c2_sources) == 1
    assert c2_sources[0]["provenance"] == "transferred"
    assert c2_sources[0]["origin_claim_id"] == "c1"
    
    # 5. Verify c2 has evidence in EvidenceIndex
    new_index: EvidenceIndex = new_ctx.get_extra(EVIDENCE_INDEX_KEY)
    assert "c2" in new_index.by_claim_id
    assert len(new_index.by_claim_id["c2"].items) == 1
    assert new_index.by_claim_id["c2"].items[0].quote == "Quote for c1"
    
    # 6. Verify c2 has evidence in global EvidencePack (ctx.evidence)
    new_evidence = new_ctx.evidence
    c2_items_in_pack = [s for s in new_evidence["items"] if s.get("claim_id") == "c2"]
    assert len(c2_items_in_pack) == 1
    assert c2_items_in_pack[0]["provenance"] == "transferred"

if __name__ == "__main__":
    pytest.main([__file__])
