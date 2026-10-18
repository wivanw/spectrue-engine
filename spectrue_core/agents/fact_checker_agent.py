# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectrue_core.domain.verification.verdict.model import AnalysisMode

from spectrue_core.utils.evidence_pack import Claim, EvidencePack
from spectrue_core.domain.evidence.model import ArticleIntent
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig
from spectrue_core.llm.llm_client import LLMClient


import logging

logger = logging.getLogger(__name__)

class FactCheckerAgent:
    """
    Main agent Facade using composed Skills.
    """
    def __init__(self, config: SpectrueConfig | None = None):
        # Lazy import to avoid circular dependency with pipeline
        from spectrue_core.agents.llm_router import LLMRouter
        from spectrue_core.adapters.llm.claims import ClaimExtractionSkill
        from spectrue_core.adapters.llm.clustering import ClusteringSkill
        from spectrue_core.adapters.llm.scoring import ScoringSkill
        from spectrue_core.adapters.llm.query import QuerySkill
        from spectrue_core.adapters.llm.article_cleaner import ArticleCleanerSkill
        from spectrue_core.adapters.llm.oracle_validation import OracleValidationSkill
        from spectrue_core.adapters.llm.relevance import RelevanceSkill
        from spectrue_core.adapters.llm.edge_typing import EdgeTypingSkill
        from spectrue_core.adapters.llm.evidence_summarizer import EvidenceSummarizerSkill
        from spectrue_core.adapters.llm.claim_judge import ClaimJudgeSkill
        
        self.config = config
        self.runtime = (config.runtime if config else None) or EngineRuntimeConfig.load_from_env()
        api_key = config.openai_api_key if config else None

        llm_concurrency = max(1, min(getattr(self.runtime.llm, "concurrency", 8), 64))

        # Create OpenAI client (Responses API)
        openai_client = LLMClient(
            openai_api_key=api_key,
            default_timeout=float(self.runtime.llm.nano_timeout_sec),
            max_retries=1,
            max_concurrent_requests=llm_concurrency,
        )

        # Create DeepSeek client (Native API compatible with Chat Completions)
        deepseek_client = None
        if self.runtime.llm.deepseek_base_url and self.runtime.llm.deepseek_api_key:
            deepseek_client = LLMClient(
                openai_api_key=self.runtime.llm.deepseek_api_key,
                base_url=self.runtime.llm.deepseek_base_url,
                default_timeout=float(self.runtime.llm.cluster_timeout_sec),
                max_retries=1,
                max_concurrent_requests=llm_concurrency,
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
        self, text: str, *, lang: str = "en", max_claims: int = 20, anchors: list | None = None, skip_enrichment: bool = False
    ) -> tuple[list[Claim], bool, ArticleIntent, str]:
        """Extract claims with article intent for Oracle triggering."""
        return await self.claims_skill.extract_claims(
            text, lang=lang, max_claims=max_claims, anchors=anchors, skip_enrichment=skip_enrichment
        )

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

    async def enrich_claims_for_planning(
        self,
        claims: list[dict],
        *,
        lang: str = "en",
        context: str | None = None,
    ) -> list[dict]:
        return await self.claims_skill.enrich_claims_for_planning(
            claims,
            lang=lang,
            context=context,
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

    async def score_evidence(self, pack: EvidencePack, *, lang: str = "en") -> dict:
        # Use default config model
        return await self.scoring_skill.score_evidence(pack, lang=lang)

    async def score_evidence_parallel(
        self, pack: EvidencePack, *, lang: str = "en", max_concurrency: int = 5
    ) -> dict:
        """Score evidence with parallel per-claim LLM calls (for deep mode)."""
        return await self.scoring_skill.score_evidence_parallel(
            pack, lang=lang, max_concurrency=max_concurrency
        )

    def detect_evidence_gaps(self, pack: EvidencePack) -> list[str]:
        return self.scoring_skill.detect_evidence_gaps(pack)

    async def analyze(self, fact: str, context: str, lang: str, analysis_mode: str | "AnalysisMode" = "general") -> dict:
        # Delegate to scoring skill for final analysis (it has the logic)
        return await self.scoring_skill.analyze(fact, context, lang, analysis_mode=analysis_mode)

    async def generate_search_queries(self, fact: str, context: str = "", lang: str = "en", content_lang: str | None = None) -> list[str]:
        return await self.query_skill.generate_search_queries(
            fact, context, lang, content_lang
        )

    async def clean_article(self, raw_text: str) -> str:
        """Clean article text using LLM Nano."""
        return await self.article_cleaner.clean_article(raw_text)

    async def verify_inline_source_relevance(
        self, 
        claims: list[dict], 
        inline_source: dict,
        article_excerpt: str = "",
        lang: str = "en",
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

        llm_reason: dict[str, str] = {}

        async def _llm_check() -> tuple[bool, bool]:
            result = await self.llm_client.call_json(
                model=self.runtime.llm.model_inline_source_verification,
                input=prompt,
                instructions="You are a source classification assistant. Distinguish primary sources (original issuers) from secondary coverage.",
                reasoning_effort="low",
                timeout=10.0,
                trace_kind="inline_source_verification",
            )
            llm_reason["reason"] = result.get("reason", "")
            return bool(result.get("is_relevant", False)), bool(result.get("is_primary", False))

        try:
            labels = None
            if self.runtime.decisions.mode("inline_source") != "off":
                # G5 (M145): Decisions is a sensor; the LLM call stays the reference path.
                from spectrue_core.adapters.llm.decision_gates import read_sensor
                from spectrue_core.agents.prompts import get_prompt
                from spectrue_core.llm.decisions import DecisionQuestion
                from spectrue_core.use_cases.retrieval.decision_gates import inline_labels_from_reading

                lang = (lang or "en")[:2]
                reading = await read_sensor(
                    gate="inline_source",
                    cfg=self.runtime.decisions,
                    llm_client=self.llm_client,
                    input_text=(
                        f'Article excerpt:\n"{article_excerpt[:500]}"\n\nClaims:\n{claims_text}\n\n'
                        f'Source:\n- URL: {url}\n- Domain: {domain}\n- Anchor: "{anchor}"'
                    ),
                    questions=[
                        DecisionQuestion.predicate("is_relevant", get_prompt(lang, "prompts.decisions.inline_source.relevant")),
                        DecisionQuestion.predicate("is_primary", get_prompt(lang, "prompts.decisions.inline_source.primary")),
                    ],
                    trace_meta={"url": url[:60], "domain": domain},
                )
                labels = inline_labels_from_reading(reading, self.runtime.calibration.decision_sensors)
            if labels is not None:
                is_relevant, is_primary = labels
                reason = "decisions_api"
            else:
                is_relevant, is_primary = await _llm_check()
                reason = llm_reason.get("reason", "")

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
