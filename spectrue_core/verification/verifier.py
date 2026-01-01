"""
FactVerifier - Main facade for fact verification.

This is the new, clean implementation that replaces the legacy FactVerifierComposite.
Uses ValidationPipeline for all verification logic.
"""

from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.pipeline import ValidationPipeline
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
import logging

logger = logging.getLogger(__name__)


class FactVerifier:
    """
    Main facade for the Fact Verification Waterfall.
    
    Uses composition with ValidationPipeline for clean architecture.
    """

    def __init__(self, config: SpectrueConfig = None, translation_service=None):
        self.config = config
        self.agent = FactCheckerAgent(config)
        # Optional translation_service for Oracle result localization
        self.pipeline = ValidationPipeline(config, self.agent, translation_service=translation_service)

    async def fetch_url_content(self, url: str) -> str | None:
        """Fetch URL content securely via configured search provider (no local requests)."""
        return await self.pipeline.search_mgr.fetch_url_content(url)

    async def verify_fact(
        self,
        fact: str,
        search_type: str = "advanced",
        gpt_model: str = "gpt-5.2",
        lang: str = "en",
        content_lang: str | None = None,
        max_cost: int | None = None,
        preloaded_context: str | None = None,
        preloaded_sources: list | None = None,
        progress_callback=None,
        needs_cleaning: bool = False,
        source_url: str | None = None,
        extract_claims_only: bool = False,  # Deep mode - just extract claims
        pipeline_profile: str | None = None,
        preloaded_claims: list | None = None, # Skip extraction if claims provided
    ) -> dict:
        """
        Execute verification via ValidationPipeline.
        """
        result = await self.pipeline.execute(
            fact=fact,
            search_type=search_type,
            gpt_model=gpt_model,
            lang=lang,
            content_lang=content_lang,
            max_cost=max_cost,
            progress_callback=progress_callback,
            preloaded_context=preloaded_context,
            preloaded_sources=preloaded_sources,
            needs_cleaning=needs_cleaning,
            source_url=source_url,
            extract_claims_only=extract_claims_only,
            pipeline_profile=pipeline_profile,
            preloaded_claims=preloaded_claims,
        )
        if "audit" not in result:
            result["audit"] = {}
        return result
