# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.verification.evidence.evidence_pack import Claim, EvidencePack, ArticleIntent
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig
from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.llm_router import LLMRouter
from spectrue_core.agents.skills.claims import ClaimExtractionSkill
from spectrue_core.agents.skills.clustering import ClusteringSkill
from spectrue_core.agents.skills.scoring import ScoringSkill
from spectrue_core.agents.skills.query import QuerySkill
from spectrue_core.agents.skills.article_cleaner import ArticleCleanerSkill
from spectrue_core.agents.skills.oracle_validation import OracleValidationSkill
from spectrue_core.agents.skills.relevance import RelevanceSkill
from spectrue_core.agents.skills.edge_typing import EdgeTypingSkill
# Per-claim judging skills (deep analysis mode)
from spectrue_core.agents.skills.evidence_summarizer import EvidenceSummarizerSkill
from spectrue_core.agents.skills.claim_judge import ClaimJudgeSkill
import logging

logger = logging.getLogger(__name__)

class FactCheckerAgent:
    """
    Main agent Facade using composed Skills.
    """
    def __init__(self, config: SpectrueConfig = None):
        self.config = config
        self.runtime = (config.runtime if config else None) or EngineRuntimeConfig.load_from_env()
        api_key = config.openai_api_key if config else None

        # Create OpenAI client (Responses API)
        openai_client = LLMClient(
            openai_api_key=api_key,
            default_timeout=float(self.runtime.llm.nano_timeout_sec),
            max_retries=3,
        )

        # Create DeepSeek client (Native API compatible with Chat Completions)
        deepseek_client = None
        if self.runtime.llm.deepseek_base_url and self.runtime.llm.deepseek_api_key:
            deepseek_client = LLMClient(
                openai_api_key=self.runtime.llm.deepseek_api_key,
                base_url=self.runtime.llm.deepseek_base_url,
                default_timeout=float(self.runtime.llm.cluster_timeout_sec),
                max_retries=3,
            )

        # Create router that directs models to appropriate clients
        self.llm_client = LLMRouter(
            openai_client=openai_client,
            chat_client=deepseek_client,
            chat_model_names=list(self.runtime.llm.deepseek_model_names) if deepseek_client else [],
        )

        # Expose LLM client for pipeline steps
        self._llm = self.llm_client

        self.claims_skill = ClaimExtractionSkill(self.config, self.llm_client)
        self.clustering_skill = ClusteringSkill(self.config, self.llm_client)
        self.scoring_skill = ScoringSkill(self.config, self.llm_client)
        self.query_skill = QuerySkill(self.config, self.llm_client)
        self.article_cleaner = ArticleCleanerSkill(self.config, self.llm_client)
        # Oracle validation skill for hybrid mode
        self.oracle_skill = OracleValidationSkill(self.config, self.llm_client)
        # Relevance skill for semantic gating
        self.relevance_skill = RelevanceSkill(self.config, self.llm_client)
        # Edge Typing for ClaimGraph C-stage
        self.edge_typing_skill = EdgeTypingSkill(self.config, self.llm_client)
        # Per-claim judging skills (deep analysis mode)
        self.evidence_summarizer_skill = EvidenceSummarizerSkill(self.llm_client)
        self.claim_judge_skill = ClaimJudgeSkill(self.llm_client)


    async def extract_claims(
        self, text: str, *, lang: str = "en", max_claims: int = 20
    ) -> tuple[list[Claim], bool, ArticleIntent, str]:
        """Extract claims with article intent for Oracle triggering."""
        return await self.claims_skill.extract_claims(text, lang=lang, max_claims=max_claims)

    async def enrich_claims_post_evidence(
        self,
        claims: list[Claim],
        *,
        lang: str = "en",
        evidence_by_claim: dict[str, list[dict]] | None = None,
    ) -> list[Claim]:
        return await self.claims_skill.enrich_claims_post_evidence(
            claims,
            lang=lang,
            evidence_by_claim=evidence_by_claim,
        )

    async def cluster_evidence(
        self,
        claims: list[Claim],
        search_results: list[dict],
        *,
        stance_pass_mode: str = "single",
    ) -> list:
        return await self.clustering_skill.cluster_evidence(
            claims,
            search_results,
            stance_pass_mode=stance_pass_mode,
        )

    async def score_evidence(self, pack: EvidencePack, *, model: str = "gpt-5.2", lang: str = "en") -> dict:
        return await self.scoring_skill.score_evidence(pack, model=model, lang=lang)

    async def score_evidence_parallel(
        self, pack: EvidencePack, *, model: str = "gpt-5.2", lang: str = "en", max_concurrency: int = 5
    ) -> dict:
        """Score evidence with parallel per-claim LLM calls (for deep mode)."""
        return await self.scoring_skill.score_evidence_parallel(
            pack, model=model, lang=lang, max_concurrency=max_concurrency
        )

    def detect_evidence_gaps(self, pack: EvidencePack) -> list[str]:
        return self.scoring_skill.detect_evidence_gaps(pack)

    async def analyze(self, fact: str, context: str, gpt_model: str, lang: str, analysis_mode: str = "general") -> dict:
        # Delegate to scoring skill for final analysis (it has the logic)
        return await self.scoring_skill.analyze(fact, context, gpt_model, lang, analysis_mode)

    async def generate_search_queries(self, fact: str, context: str = "", lang: str = "en", content_lang: str = None) -> list[str]:
        return await self.query_skill.generate_search_queries(
            fact, context, lang, content_lang
        )

    async def clean_article(self, raw_text: str) -> str:
        """Clean article text using LLM Nano."""
        return await self.article_cleaner.clean_article(raw_text)

    async def verify_search_relevance(self, claims: list[Claim], search_results: list[dict]) -> dict:
        """Verify if search results are semantically relevant to claims."""
        return await self.relevance_skill.verify_search_relevance(claims, search_results)

    async def evaluate_semantic_gating(self, claims: list) -> bool:
        """
        Evaluate semantic gating policy (e.g. reject unverifiable content).
        Returns True if allowed, False if rejected.
        """
        # Delegate to relevance skill or scoring skill if available. 
        # For now, default to True (allow) unless filtering logic is restored/implemented.
        # Tests will mock this to return False.
        if hasattr(self.relevance_skill, "evaluate_semantic_gating"):
             return await self.relevance_skill.evaluate_semantic_gating(claims)
        return True


    async def verify_oracle_relevance(self, user_fact: str, oracle_claim: str, oracle_rating: str) -> bool:
        """
        Check if the Oracle result is semantically relevant to the user's fact.
        Prevents false positives where keywords match but the topic differs (e.g. comet news vs comet alien fake).
        """
        # Quick length check
        if not user_fact or not oracle_claim:
            return False

        prompt = f"""Compare the User Query with the Fact-Check Result.
        
User Query: "{user_fact[:1000]}"
Fact-Check Claim: "{oracle_claim[:1000]}" (Rated: {oracle_rating})

Task:
Determine if the Fact-Check is discussing the SAME specific claim or event as the User Query.
- If the User Query is about a normal event (e.g. "comet approaching Earth") and the Fact-Check is debunking a wild/fake story about it (e.g. "aliens found on comet"), they are DIFFERENT (Relevance: NO).
- If the User Query IS the fake story (e.g. "aliens on comet?"), then it IS RELEVANT (Relevance: YES).
- If the User Query is vague, but the Fact-Check is the primary context, it's relevant.

Output JSON: {{ "is_relevant": true/false, "reason": "..." }}
"""
        try:
            from spectrue_core.utils.trace import Trace
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a relevance checking assistant.",
                reasoning_effort="low",
                timeout=10.0,
                trace_kind="oracle_verification"
            )
            is_relevant = bool(result.get("is_relevant", False))
            Trace.event("oracle.verification", {
                "user_fact": user_fact[:50], 
                "oracle_claim": oracle_claim[:50], 
                "is_relevant": is_relevant
            })
            if not is_relevant:
                 logger.debug("[Agent] Oracle hit rejected by LLM: %s", result.get("reason"))
            return is_relevant

        except Exception as e:
            logger.warning("[Agent] Oracle verification failed: %s. Assuming relevant.", e)
            return True

    async def verify_inline_source_relevance(
        self, 
        claims: list[dict], 
        inline_source: dict,
        article_excerpt: str = ""
    ) -> dict:
        """
        Check if an inline source is relevant to the extracted claims.
        
        T7: Inline sources are URLs found in article text that reference external sources.
        They may be primary sources (e.g., official statement being quoted) or just
        contextual links (e.g., author's social media).
        
        Args:
            claims: List of extracted claims from the article
            inline_source: Source dict with 'url', 'title', 'domain' keys
            article_excerpt: First ~500 chars of article for context
            
        Returns:
            dict with 'is_relevant', 'is_primary', 'reason', 'verification_skipped' keys
        """
        from spectrue_core.utils.trace import Trace

        # Short-circuit: if inline source verification is disabled
        if not self.runtime.llm.enable_inline_source_verification:
            return {
                "is_relevant": True,
                "is_primary": False,
                "reason": "inline_verification_disabled",
                "verification_skipped": True,
            }

        if not claims or not inline_source:
            return {"is_relevant": False, "is_primary": False, "reason": "empty_input"}

        url = inline_source.get("url", "")
        domain = inline_source.get("domain", "")
        anchor = inline_source.get("title", "") or inline_source.get("anchor", "")

        # Format claims for prompt
        claims_text = "\n".join([
            f"- {c.get('text', '')}" for c in claims[:5]
        ])

        # Simplified prompt without if/then rules
        prompt = f"""Analyze if this source URL is a PRIMARY SOURCE for the claims.

Article Excerpt:
"{article_excerpt[:500]}"

Claims:
{claims_text}

Source:
- URL: {url}
- Domain: {domain}
- Anchor: "{anchor}"

Evaluate:
1. Is this source RELEVANT to any claim?
2. Is this source PRIMARY (the original issuer of the information) or SECONDARY (coverage/reporting about it)?

A PRIMARY source is where the information originated (official statement, original document, author's own platform).
A SECONDARY source reports or discusses information from elsewhere.

Output JSON: {{ "is_relevant": true/false, "is_primary": true/false, "reason": "one sentence explanation" }}
"""

        try:
            result = await self.llm_client.call_json(
                model=self.runtime.llm.model_inline_source_verification,
                input=prompt,
                instructions="You are a source classification assistant. Distinguish primary sources (original issuers) from secondary coverage.",
                reasoning_effort="low",
                timeout=10.0,
                trace_kind="inline_source_verification",
            )

            is_relevant = bool(result.get("is_relevant", False))
            is_primary = bool(result.get("is_primary", False))
            reason = result.get("reason", "")

            Trace.event("inline_source.verification", {
                "url": url[:60],
                "domain": domain,
                "is_relevant": is_relevant,
                "is_primary": is_primary,
                "reason": reason
            })

            if is_primary:
                logger.debug("[Agent] Inline source PRIMARY: %s", domain)
            elif not is_relevant:
                logger.debug("[Agent] Inline source rejected: %s - %s", domain, reason[:60])

            return {
                "is_relevant": is_relevant,
                "is_primary": is_primary,
                "reason": reason
            }

        except Exception as e:
            logger.warning("[Agent] Inline source verification failed: %s. Marking as secondary.", e)
            return {"is_relevant": True, "is_primary": False, "reason": f"verification_error: {e}"}
