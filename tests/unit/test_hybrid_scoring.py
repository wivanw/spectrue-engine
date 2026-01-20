# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

import pytest
from unittest.mock import MagicMock
from spectrue_core.agents.skills.scoring import ScoringSkill
from spectrue_core.llm.model_registry import ModelID

@pytest.mark.unit
class TestHybridScoring:
    
    @pytest.fixture
    def skill(self, mock_llm_client, mock_config):
        mock_config.openai_model = ModelID.PRO
        # Mocking runtime for ScoringSkill
        mock_config.runtime = MagicMock()
        mock_config.runtime.llm = MagicMock()
        mock_config.runtime.llm.timeout_sec = 30
        mock_config.runtime.llm.judge_out_tokens_estimate = 400
        mock_config.runtime.llm.deepseek_fail_prob = 0.0
        
        skill = ScoringSkill(config=mock_config, llm_client=mock_llm_client)
        return skill

    @pytest.mark.asyncio
    async def test_hybrid_score_calculation(self, skill, mock_llm_client):
        # Evidence G = 0.8, Prior = 1.0
        # Bayesian update: logit(0.8) + 0.2 * logit(1.0)
        # logit(0.8) ~ 1.386, logit(1.0) ~ 20.7 (clamped)
        # L = 1.386 + 4.14 = 5.52 -> sigmoid(5.52) ~ 0.996
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {
                    "claim_id": "c1", 
                    "verdict_score": 0.8, 
                    "verdict": "verified",
                    "reason": "Supported by evidence",
                    "rgba": [0.1, 0.8, 0.9, 0.9],
                    "prior_score": 1.0,
                    "prior_reason": "General knowledge"
                }
            ],
            "verified_score": 0.5,
            "explainability_score": 0.9,
            "danger_score": 0.1,
            "style_score": 0.9,
            "rationale": "Global rationale"
        }
        
        pack = {
            "claims": [{"id": "c1", "importance": 1.0}],
            "search_results": [{"claim_id": "c1", "relevance_score": 0.9}],
            "scored_sources": [{"claim_id": "c1", "stance": "SUPPORT"}]
        }
        
        result = await skill.score_evidence(pack)
        
        # Check re-aggregated verified_score (Bayesian hybrid)
        assert result["verified_score"] > 0.99
        assert result["claim_verdicts"][0]["rgba"][1] > 0.99

    @pytest.mark.asyncio
    async def test_hybrid_score_unverified_anchor_blocks(self, skill, mock_llm_client):
        # Even if prior is high, if evidence G is -1.0, total is -1.0
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {
                    "claim_id": "c1", 
                    "verdict_score": -1.0, 
                    "verdict": "unverified",
                    "reason": "No evidence",
                    "rgba": [0.1, -1.0, 0.9, 0.0],
                    "prior_score": 1.0,
                    "prior_reason": "I know it's true but no evidence"
                }
            ],
            "verified_score": 0.5,
            "explainability_score": 0.0,
            "danger_score": 0.1,
            "style_score": 0.9,
            "rationale": "Test"
        }
        
        pack = {
            "claims": [{"id": "c1", "importance": 1.0}],
            "search_results": [],
        }
        
        result = await skill.score_evidence(pack)
        assert result["verified_score"] == -1.0

    @pytest.mark.asyncio
    async def test_prior_unknown_sentinel_is_ignored(self, skill, mock_llm_client):
        # G = 0.8, Prior = -1.0 -> Should be 0.8 (Prior ignored)
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {
                    "claim_id": "c1", 
                    "verdict_score": 0.8, 
                    "verdict": "verified",
                    "reason": "Supported",
                    "rgba": [0.1, 0.8, 0.9, 0.9],
                    "prior_score": -1.0,
                    "prior_reason": "NEI"
                }
            ],
            "verified_score": 0.5,
            "explainability_score": 0.9,
            "danger_score": 0.1,
            "style_score": 0.9,
            "rationale": "Test"
        }
        
        pack = {
            "claims": [{"id": "c1", "importance": 1.0}],
            "search_results": [{"claim_id": "c1"}],
        }
        
        result = await skill.score_evidence(pack)
        assert result["verified_score"] == 0.8

    @pytest.mark.asyncio
    async def test_neutral_prior_05_results_in_G(self, skill, mock_llm_client):
        # G = 0.8, Prior = 0.5 -> Should be 0.8 (Prior is 0 log-odds)
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {
                    "claim_id": "c1", 
                    "verdict_score": 0.8, 
                    "verdict": "verified",
                    "reason": "Supported",
                    "rgba": [0.1, 0.8, 0.9, 0.9],
                    "prior_score": 0.5,
                    "prior_reason": "Neutral scientific consensus"
                }
            ],
            "verified_score": 0.5,
            "explainability_score": 0.9,
            "danger_score": 0.1,
            "style_score": 0.9,
            "rationale": "Test"
        }
        
        pack = {
            "claims": [{"id": "c1", "importance": 1.0}],
            "search_results": [{"claim_id": "c1"}],
        }
        
        result = await skill.score_evidence(pack)
        assert result["verified_score"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_nei_prior_reason_is_ignored(self, skill, mock_llm_client):
        # G = 0.8, Prior = 1.0 BUT reason has NEI -> Should be 0.8
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {
                    "claim_id": "c1", 
                    "verdict_score": 0.8, 
                    "verdict": "verified",
                    "reason": "Supported",
                    "rgba": [0.1, 0.8, 0.9, 0.9],
                    "prior_score": 1.0,
                    "prior_reason": "NEI: not in my training data"
                }
            ],
            "verified_score": 0.5,
            "explainability_score": 0.9,
            "danger_score": 0.1,
            "style_score": 0.9,
            "rationale": "Test"
        }
        
        pack = {
            "claims": [{"id": "c1", "importance": 1.0}],
            "search_results": [{"claim_id": "c1"}],
        }
        
        result = await skill.score_evidence(pack)
        assert result["verified_score"] == 0.8

    @pytest.mark.asyncio
    async def test_prior_influences_low_veracity(self, skill, mock_llm_client):
        # G = 0.2, Prior = 0.8
        # L_g = -1.386, L_prior = 1.386
        # L_f = -1.386 + 0.2*1.386 = -1.109 -> sigmoid(-1.109) = 0.248
        mock_llm_client.call_json.return_value = {
            "claim_verdicts": [
                {
                    "claim_id": "c1", 
                    "verdict_score": 0.2, 
                    "verdict": "unlikely",
                    "reason": "Weak evidence",
                    "rgba": [0.3, 0.2, 0.9, 0.4],
                    "prior_score": 0.8,
                    "prior_reason": "Probably true based on general knowledge"
                }
            ],
            "verified_score": 0.5,
            "explainability_score": 0.4,
            "danger_score": 0.1,
            "style_score": 0.9,
            "rationale": "Test"
        }
        
        pack = {
            "claims": [{"id": "c1", "importance": 1.0}],
            "search_results": [{"claim_id": "c1"}],
        }
        
        result = await skill.score_evidence(pack)
        assert result["verified_score"] == pytest.approx(0.248, abs=0.001)
