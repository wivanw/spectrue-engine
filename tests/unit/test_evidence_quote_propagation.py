import pytest
from spectrue_core.verification.evidence.evidence import build_evidence_pack
from spectrue_core.verification.evidence.evidence_stance import assign_claim_rgba
from spectrue_core.pipeline.mode import ScoringMode

def test_quote_propagation_to_rgba():
    """
    Test that a quote in the raw source successfully propagates to EvidenceItem
    and LLM A-score passes through unchanged (M133: cap removed).
    """
    fact = "Кристали мають впорядковану структуру."
    claims = [
        {"id": "c1", "text": fact, "normalized_text": fact, "importance": 1.0}
    ]
    
    # Raw source with a quote
    sources = [{
        "url": "https://example.com/crystal",
        "domain": "example.com",
        "title": "Crystal Science",
        "snippet": "Crystals have a structure.",
        "quote": "Crystals are characterized by a highly ordered arrangement of atoms.",
        "stance": "support",
        "relevance_score": 0.9,
        "evidence_tier": "A"
    }]
    
    # 1. Build pack
    pack = build_evidence_pack(
        fact=fact,
        claims=claims, # type: ignore
        sources=sources
    )
    
    # Verify search_results has quote
    assert pack["search_results"][0]["quote"] == "Crystals are characterized by a highly ordered arrangement of atoms."
    
    # Verify items have quote
    assert pack["items"][0]["quote"] == "Crystals are characterized by a highly ordered arrangement of atoms."
    
    # 2. Assign RGBA
    verdict = {
        "claim_id": "c1",
        "verdict_score": 0.9,
        "rgba": [0.0, 0.9, 0.9, 0.9] # LLM says 0.9
    }
    
    assign_claim_rgba(
        verdict,
        global_r=0.0,
        global_b=0.9,
        global_a=0.9,
        judge_mode=ScoringMode.STANDARD,
        pack=pack
    )
    
    # M133: Alpha cap removed — LLM A-score passes through unchanged
    final_a = verdict["rgba"][3]
    # LLM said 0.9, so final should be 0.9 (no cap)
    assert final_a == pytest.approx(0.9, abs=0.01)

if __name__ == "__main__":
    pytest.main([__file__])

