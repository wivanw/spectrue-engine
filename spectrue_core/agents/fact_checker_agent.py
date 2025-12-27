from spectrue_core.verification.evidence_pack import Claim, EvidencePack, ArticleIntent
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig
from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.agents.skills.claims import ClaimExtractionSkill
from spectrue_core.agents.skills.clustering import ClusteringSkill
from spectrue_core.agents.skills.scoring import ScoringSkill
from spectrue_core.agents.skills.query import QuerySkill
from spectrue_core.agents.skills.article_cleaner import ArticleCleanerSkill
from spectrue_core.agents.skills.oracle_validation import OracleValidationSkill
from spectrue_core.agents.skills.relevance import RelevanceSkill
from spectrue_core.agents.skills.edge_typing import EdgeTypingSkill
from spectrue_core.verification.inline_verification import InlineVerificationSkill
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
        # M63: Oracle validation skill for hybrid mode
        self.oracle_skill = OracleValidationSkill(self.config, self.llm_client)
        # M66: Relevance skill for semantic gating
        self.relevance_skill = RelevanceSkill(self.config, self.llm_client)
        # M67: Inline Verification (Social Identity check)
        self.inline_verification_skill = InlineVerificationSkill(self.config, self.llm_client)
        # M72: Edge Typing for ClaimGraph C-stage
        self.edge_typing_skill = EdgeTypingSkill(self.config, self.llm_client)

    async def extract_claims(
        self, text: str, *, lang: str = "en", max_claims: int = 7
    ) -> tuple[list[Claim], bool, ArticleIntent]:
        """Extract claims with article intent for M63 Oracle triggering."""
        return await self.claims_skill.extract_claims(text, lang=lang, max_claims=max_claims)

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
        """M66: Verify if search results are semantically relevant to claims."""
        return await self.relevance_skill.verify_search_relevance(claims, search_results)
        
    async def verify_social_statement(self, claim: Claim, snippet: str, url: str) -> dict:
        """M67: Verify Key Official Statement from Social Media (Tier A')."""
        return await self.inline_verification_skill.verify_social_statement(claim, snippet, url)

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
            dict with 'is_relevant', 'is_primary', 'reason' keys
        """
        from spectrue_core.utils.trace import Trace
        
        if not claims or not inline_source:
            return {"is_relevant": False, "is_primary": False, "reason": "empty_input"}
        
        url = inline_source.get("url", "")
        domain = inline_source.get("domain", "")
        anchor = inline_source.get("title", "") or inline_source.get("anchor", "")
        
        # Format claims for prompt
        claims_text = "\n".join([
            f"- {c.get('text', '')}" for c in claims[:5]
        ])
        
        # Detect "X said/announced" pattern for auto-primary rule
        # If claim mentions someone saying something and URL is their platform, it's primary
        prompt = f"""Analyze if this source URL is relevant as PRIMARY EVIDENCE for the claims.

Article Excerpt (for context):
"{article_excerpt[:500]}"

Extracted Claims:
{claims_text}

Inline Source Found:
- URL: {url}
- Domain: {domain}
- Anchor text: "{anchor}"

Task:
1. Determine if this source is RELEVANT to any of the claims
2. Determine if this is a PRIMARY SOURCE (the original source being quoted/referenced)

Rules for PRIMARY source detection:
- If a claim says "X announced/said/posted" and the URL is X's official platform (their official site, social media), it's PRIMARY
- If the URL is the original document/statement being cited, it's PRIMARY
- News articles about a topic are NOT primary (they are secondary coverage)
- Social media of the article author is NOT primary (it's just metadata)

Output JSON: {{ "is_relevant": true/false, "is_primary": true/false, "reason": "brief explanation" }}
"""
        
        try:
            result = await self.llm_client.call_json(
                model="gpt-5-nano",
                input=prompt,
                instructions="You are a source relevance analyzer for fact-checking.",
                reasoning_effort="low",
                timeout=10.0,
                trace_kind="inline_source_verification"
            )
            
            is_relevant = bool(result.get("is_relevant", False))
            is_primary = bool(result.get("is_primary", False))
            reason = result.get("reason", "")
            
            Trace.event("inline_source.verification", {
                "url": url[:60],
                "domain": domain,
                "is_relevant": is_relevant,
                "is_primary": is_primary,
                "reason": reason  # Full reason for debugging
            })
            
            if is_primary:
                logger.debug("[Agent] Inline source PRIMARY: %s", domain)
            elif not is_relevant:
                # Full reason available in trace, keep console clean
                logger.debug("[Agent] Inline source rejected: %s - %s", domain, reason[:60])
            
            return {
                "is_relevant": is_relevant,
                "is_primary": is_primary,
                "reason": reason
            }
            
        except Exception as e:
            logger.warning("[Agent] Inline source verification failed: %s. Marking as secondary.", e)
            return {"is_relevant": True, "is_primary": False, "reason": f"verification_error: {e}"}
