from spectrue_core.verification.search_mgr import SearchManager, SEARCH_COSTS
from spectrue_core.verification.evidence import build_evidence_pack
from spectrue_core.verification.trusted_sources import get_trusted_domains_by_lang
from spectrue_core.utils.text_processing import clean_article_text, normalize_search_query
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.utils.trust_utils import enrich_sources_with_trust
from spectrue_core.utils.trace import Trace
from spectrue_core.config import SpectrueConfig
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
import logging
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ValidationPipeline:
    """
    Orchestrates the fact-checking waterfall process.
    """
    def __init__(self, config: SpectrueConfig, agent: FactCheckerAgent):
        self.config = config
        self.agent = agent
        self.search_mgr = SearchManager(config)

    async def execute(
        self,
        fact: str,
        search_type: str,
        gpt_model: str,
        lang: str,
        content_lang: str | None = None,
        max_cost: int | None = None,
        progress_callback=None,
        preloaded_context: str | None = None,
        preloaded_sources: list | None = None,
        include_internal: bool = False,
    ) -> dict:
        
        # 1. Initialize
        self.search_mgr.reset_metrics()
        final_context = preloaded_context or ""
        final_sources = preloaded_sources or []
        
        # 2. URL Pre-processing (M47)
        original_fact = fact
        if self._is_url(fact) and not preloaded_context:
            fetched_text = await self._resolve_url_content(fact)
            if fetched_text:
                fact = fetched_text  # Update fact to be the content of the page
                final_context = fetched_text

        # 3. Claims Extraction
        if progress_callback: await progress_callback("extracting_claims")
        
        cleaned_fact = clean_article_text(fact)
        claims = await self.agent.extract_claims(cleaned_fact[:4000], lang=lang)
        
        # Generate Queries
        search_queries = self._generate_initial_queries(claims, fact)
        
        # 4. Oracle Check
        if progress_callback: await progress_callback("checking_oracle")
        oracle_res = await self.search_mgr.check_oracle(search_queries[0], content_lang or lang)
        
        if oracle_res:
             # Fast exit
             logger.info("[Pipeline] Oracle hit. Stopping.")
             return self._finalize_oracle(oracle_res, original_fact, max_cost, include_internal)

        # 5. Waterfall Search (Tier 1 -> Tier 2 -> CSE)
        if not preloaded_context:
            if progress_callback: await progress_callback("searching_tier1")
            
            # Tier 1
            search_lang = content_lang or lang
            tier1_domains = get_trusted_domains_by_lang(search_lang)
            # Use query[1] (content lang) if available, else query[0]
            t1_query = search_queries[1] if len(search_queries) > 1 else search_queries[0]
            
            # Check budget
            if self._can_add_search(gpt_model, search_type, max_cost):
                ctx, srcs = await self.search_mgr.search_tier1(t1_query, tier1_domains)
                final_context += "\n" + ctx
                final_sources.extend(srcs)
            
            # Tier 2: General / Deep Dive
            # Trigger if:
            # 1. Explicitly requested (search_type="advanced" or "deep")
            # 2. Tier 1 yielded weak results (< 2 sources) AND budget allows
            
            should_run_tier2 = (search_type in ["advanced", "deep"]) or (len(final_sources) < 2)
            
            if should_run_tier2 and self._can_add_search(gpt_model, search_type, max_cost):
                if progress_callback: await progress_callback("searching_tier2")
                
                # Exclude domains we already visited in Tier 1
                visited_domains = [get_registrable_domain(s.get("url", "")) for s in final_sources]
                
                # Use query[0] (English/Global) for broader reach if available, else local
                t2_query = search_queries[0]
                
                ctx, srcs = await self.search_mgr.search_tier2(t2_query, exclude_domains=visited_domains)
                if srcs:
                    final_context += "\n\n=== GENERAL SEARCH ===\n" + ctx
                    final_sources.extend(srcs)
                else:
                    # Tier 3: Google CSE Fallback (if Tavily failed completely)
                    if self.search_mgr.tavily_calls > 0 and len(final_sources) == 0:
                         if progress_callback: await progress_callback("searching_cse")
                         cse_results = await self.search_mgr.search_google_cse(t1_query, lang=lang)
                         # Convert CSE to standard source format
                         for res in cse_results:
                                final_sources.append({
                                    "url": res.get("link"),
                                    "domain": get_registrable_domain(res.get("link")),
                                    "title": res.get("title"),
                                    "content": res.get("snippet", ""),
                                    "source_type": "general", # CSE doesn't classify well
                                    "is_trusted": False
                                })

        # 6. Analysis and Scoring
        if progress_callback: await progress_callback("ai_analysis")
        
        current_cost = self.search_mgr.calculate_cost(gpt_model, search_type)
        
        # Cluster (T168)
        clustered_results = None
        if claims and final_sources:
             clustered_results = await self.agent.cluster_evidence(claims, final_sources, lang=lang)

        # Build Pack
        pack = build_evidence_pack(
            fact=original_fact, # Use original fact/url as the anchor
            claims=claims,
            sources=final_sources,
            search_results_clustered=clustered_results,
            content_lang=content_lang or lang,
            article_context={"text_excerpt": fact[:500]} if fact != original_fact else None
        )
        
        # Score (T164)
        result = await self.agent.score_evidence(pack, model=gpt_model, lang=lang)
        
        # Finalize
        result["cost"] = current_cost
        result["text"] = original_fact
        result["search_meta"] = self.search_mgr.get_search_meta()
        result["sources"] = enrich_sources_with_trust(final_sources)
        
        # Cap enforcement
        global_cap = pack.get("constraints", {}).get("global_cap", 1.0)
        verified = result.get("verified_score", 0.5)
        if verified > global_cap:
            result["verified_score"] = global_cap
            result["cap_applied"] = True
        
        return result

    def _is_url(self, text: str) -> bool:
        return text and ("http://" in text or "https://" in text) and len(text) < 500

    async def _resolve_url_content(self, url: str) -> str | None:
        try:
             # Basic cleanup similar to original
             text = await self.search_mgr.fetch_url_content(url)
             # ... cleanup logic ...
             return text
        except Exception:
            return None

    def _generate_initial_queries(self, claims: list, fact: str) -> list[str]:
        # Collect from claims
        qs = []
        for c in claims:
            if c.get("search_queries"):
                qs.extend(c["search_queries"])
        if not qs:
            qs = [fact[:200]]
        
        # Normalize
        normalized = [normalize_search_query(q) for q in qs]
        # Dedupe
        seen = set()
        final = []
        for q in normalized:
            if q not in seen:
                seen.add(q)
                final.append(q)
        return final[:2]

    def _can_add_search(self, model, search_type, max_cost):
        # Speculative cost check
        current = self.search_mgr.calculate_cost(model, search_type)
        # Add cost of 1 search
        step_cost = int(SEARCH_COSTS.get(search_type, 80))
        return self.search_mgr.can_afford(current + step_cost, max_cost)

    def _finalize_oracle(self, oracle_res, fact, max_cost, include_internal):
        # ... logic to format oracle result ...
        oracle_res["text"] = fact
        oracle_res["search_meta"] = self.search_mgr.get_search_meta()
        return oracle_res

