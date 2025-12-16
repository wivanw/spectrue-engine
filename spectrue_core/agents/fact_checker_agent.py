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
                 logger.info("[Agent] Oracle hit rejected by LLM: %s", result.get("reason"))
            return is_relevant
            
        except Exception as e:
            logger.warning("[Agent] Oracle verification failed: %s. Assuming relevant.", e)
            return True