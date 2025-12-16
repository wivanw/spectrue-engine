from spectrue_core.verification.evidence_pack import Claim, EvidencePack
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig
from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.skills.claims import ClaimExtractionSkill
from spectrue_core.agents.skills.clustering import ClusteringSkill
from spectrue_core.agents.skills.scoring import ScoringSkill
from spectrue_core.agents.skills.query import QuerySkill
from spectrue_core.agents.skills.article_cleaner import ArticleCleanerSkill
import logging
import asyncio

logger = logging.getLogger(__name__)

class FactCheckerAgent:
    """
    Main agent Facade using composed Skills.
    """
    def __init__(self, config: SpectrueConfig = None):
        self.config = config
        self.runtime = (config.runtime if config else None) or EngineRuntimeConfig.load_from_env()
        api_key = config.openai_api_key if config else None
        
        self.llm_client = LLMClient(
            openai_api_key=api_key,
            default_timeout=float(self.runtime.llm.nano_timeout_sec),
            max_retries=3,
        )
        
        self.claims_skill = ClaimExtractionSkill(self.config, self.llm_client)
        self.clustering_skill = ClusteringSkill(self.config, self.llm_client)
        self.scoring_skill = ScoringSkill(self.config, self.llm_client)
        self.query_skill = QuerySkill(self.config, self.llm_client)
        self.article_cleaner = ArticleCleanerSkill(self.config, self.llm_client)

    async def extract_claims(self, text: str, *, lang: str = "en", max_claims: int = 7) -> list[Claim]:
        return await self.claims_skill.extract_claims(text, lang=lang, max_claims=max_claims)

    async def cluster_evidence(self, claims: list[Claim], search_results: list[dict], *, lang: str = "en") -> list:
        return await self.clustering_skill.cluster_evidence(claims, search_results, lang=lang)

    async def score_evidence(self, pack: EvidencePack, *, model: str = "gpt-5.2", lang: str = "en") -> dict:
        return await self.scoring_skill.score_evidence(pack, model=model, lang=lang)

    def detect_evidence_gaps(self, pack: EvidencePack) -> list[str]:
        return self.scoring_skill.detect_evidence_gaps(pack)
    
    async def analyze(self, fact: str, context: str, gpt_model: str, lang: str, analysis_mode: str = "general") -> dict:
        # Delegate to scoring skill for final analysis (it has the logic)
        return await self.scoring_skill.analyze(fact, context, gpt_model, lang, analysis_mode)

    async def generate_search_queries(self, fact: str, context: str = "", lang: str = "en", content_lang: str = None, *, allow_short_llm: bool = False) -> list[str]:
        return await self.query_skill.generate_search_queries(
            fact, context, lang, content_lang, allow_short_llm=allow_short_llm
        )
    
    async def clean_article(self, raw_text: str) -> str:
        """Clean article text using LLM Nano."""
        return await self.article_cleaner.clean_article(raw_text)