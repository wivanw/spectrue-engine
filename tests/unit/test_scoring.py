
import pytest
from spectrue_core.agents.skills.scoring import ScoringSkill

@pytest.mark.unit
class TestScoringSkill:
    
    @pytest.fixture
    def skill(self, mock_llm_client, mock_config):
        # BaseSkill(config: SpectrueConfig, llm_client: LLMClient)
        skill = ScoringSkill(config=mock_config, llm_client=mock_llm_client)
        return skill

    @pytest.mark.asyncio
    async def test_score_evidence_basic(self, skill, mock_llm_client):
        # Mock LLM response with claim verdicts
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {"claim_id": "c1", "verdict_score": 0.8}
            ],
            "danger_score": 0.1,
            "style_score": 0.9,
            "explainability_score": 0.9,
            "rationale": "Solid evidence."
        }
        
        # Provide claim in pack so importance weighting works
        # M62: Include metrics with sufficient independent_domains to avoid hard caps
        pack = {
            "original_fact": "Test", 
            "search_results": [],
            "claims": [{"id": "c1", "importance": 1.0, "type": "core"}],
            "metrics": {
                "per_claim": {
                    "c1": {"independent_domains": 3, "primary_present": True}
                }
            }
        }
        result = await skill.score_evidence(pack)
        
        assert result["verified_score"] == 0.8
        assert result["rationale"] == "Solid evidence."

    @pytest.mark.asyncio
    async def test_score_capping(self, skill, mock_llm_client):
        # Pack has global cap 0.6
        pack = {
            "original_fact": "Test", 
            "constraints": {"global_cap": 0.6},
            "search_results": [],
            "claims": [{"id": "c1", "importance": 1.0, "type": "core"}]
        }
        
        # LLM tries to give 0.9 via claim verdict
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {"claim_id": "c1", "verdict_score": 0.9}
            ],
            "rationale": "It is true."
        }
        
        result = await skill.score_evidence(pack)
        
        # Should be clamped
        assert result["verified_score"] == 0.6
        assert result.get("cap_enforced") is True

    def test_strip_internal_source_markers(self, skill):
        # Test cleaning of [TRUSTED], [REL=0.9], [RAW]
        text = "Fact is true according to [TRUSTED] source [REL=0.99] CNN."
        cleaned = skill._strip_internal_source_markers(text)
        assert cleaned == "Fact is true according to source CNN."
        
        text2 = "[RAW] [REL=0.5] Some text."
        cleaned2 = skill._strip_internal_source_markers(text2)
        assert cleaned2 == "Some text."

    def test_maybe_drop_style_section_low_honesty(self, skill):
        # If honesty < 0.8, should keep style section?
        # Code: "if h is None or h < 0.80: return rationale" -> KEEPS IT
        
        rationale = "Main text.\n\nStyle and Context:\nBiased."
        # Honesty 0.5 -> Keep
        res = skill._maybe_drop_style_section(rationale, honesty_score=0.5, lang="en")
        assert "Style and Context:" in res
        
        # Honesty 0.9 -> Drop
        res2 = skill._maybe_drop_style_section(rationale, honesty_score=0.9, lang="en")
        assert "Style and Context:" not in res2
        assert "Main text." in res2
        assert "Biased" not in res2

    @pytest.mark.asyncio
    async def test_analyze_flow(self, skill, mock_llm_client):
        # Test analyze() post-processing
        mock_llm_client.call_json.return_value = {
            "verified_score": 0.5,
            "rationale": "Some [TRUSTED] rationale.",
            "style_score": 0.9,
            "context_score": 0.9
        }
        
        res = await skill.analyze("fact", "context", "gpt-5", "en")
        
        # Check stripping markers
        assert "[TRUSTED]" not in res["rationale"]
        
        # Check style drop (honesty = (0.9+0.9)/2 = 0.9 > 0.8)
        # Note: logic inside analyze() also calls _strip_internal_source_markers AND _maybe_drop_style_section
        assert res["verified_score"] == 0.5
