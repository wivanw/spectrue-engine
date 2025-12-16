from spectrue_core.verification.trusted_sources import (
    TRUSTED_SOURCES, AVAILABLE_TOPICS
)
from spectrue_core.config import SpectrueConfig
import logging
from .legacy.fact_verifier_composite import FactVerifierComposite

from .pipeline import ValidationPipeline
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent

logger = logging.getLogger(__name__)

class FactVerifier(FactVerifierComposite):
    """
    Main facade for the Fact Verification Waterfall.
    
    Refactors the old FactVerifierComposite into a cleaner structure using composition
    instead of inheritance/monolith, while maintaining backward compatibility
    by inheriting from the legacy class until fully migrated.
    """
    
    def __init__(self, config: SpectrueConfig = None):
        super().__init__(config)
        self.agent = FactCheckerAgent(config)
        self.pipeline = ValidationPipeline(config, self.agent)

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
        include_internal: bool = False,
        progress_callback=None,
    ) -> dict:
        """
        Execute verification via the new ValidationPipeline.
        """
        return await self.pipeline.execute(
            fact=fact,
            search_type=search_type,
            gpt_model=gpt_model,
            lang=lang,
            content_lang=content_lang,
            max_cost=max_cost,
            progress_callback=progress_callback,
            preloaded_context=preloaded_context,
            preloaded_sources=preloaded_sources,
            include_internal=include_internal
        )
