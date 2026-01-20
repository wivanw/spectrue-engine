from spectrue_core.llm.model_registry import ModelID
# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.


import pytest
from spectrue_core.agents.skills.scoring import ScoringSkill

@pytest.mark.unit
class TestScoringSkill:
    
    @pytest.fixture
    def skill(self, mock_llm_client, mock_config):
        # BaseSkill(config: SpectrueConfig, llm_client: LLMClient)
        mock_config.openai_model = ModelID.PRO
        skill = ScoringSkill(config=mock_config, llm_client=mock_llm_client)
        return skill

    @pytest.mark.asyncio
    async def test_score_evidence_llm_aggregation(self, skill, mock_llm_client):
        """v2: LLM provides verified_score directly."""
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {"claim_id": "c1", "verdict_score": 0.8}
            ],
            "verified_score": 0.85,  # LLM provides this now!
            "explainability_score": 0.7,
            "danger_score": 0.1,
            "style_score": 0.9,
            "rationale": "Solid evidence."
        }
        
        pack = {
            "original_fact": "Test", 
            "search_results": [
                {"claim_id": "c1", "domain": "reuters.com", "is_trusted": True}
            ],
            "claims": [{"id": "c1", "importance": 1.0}],
        }
        result = await skill.score_evidence(pack)
        
        # LLM's verified_score is ignored in favor of re-aggregation from claims
        assert result["verified_score"] == 0.8
        assert result["rationale"] == "Solid evidence."

    @pytest.mark.asyncio
    async def test_fallback_when_llm_forgets_verified_score(self, skill, mock_llm_client):
        """v2: Fallback to mean if LLM forgets verified_score."""
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {"claim_id": "c1", "verdict_score": 0.8},
                {"claim_id": "c2", "verdict_score": 0.6}
            ],
            # verified_score missing!
            "explainability_score": 0.6,
            "danger_score": 0.1,
            "style_score": 0.9,
            "rationale": "Analysis."
        }
        
        pack = {
            "original_fact": "Test", 
            "search_results": [],
            "claims": [
                {"id": "c1", "importance": 0.9},
                {"id": "c2", "importance": 0.3}
            ]
        }
        
        result = await skill.score_evidence(pack)
        
        # M111+: verified_score is now anchor-based, not averaged.
        # Parsing returns -1.0 sentinel when LLM doesn't provide a score.
        # Actual G is computed later from anchor claim.
        assert result["verified_score"] == -1.0

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
            "explainability_score": 0.8,
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
