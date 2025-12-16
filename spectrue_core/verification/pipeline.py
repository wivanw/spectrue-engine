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
import asyncio
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
        if progress_callback: await progress_callback("analyzing_input")
        
        self.search_mgr.reset_metrics()
        final_context = preloaded_context or ""
        final_sources = preloaded_sources or []
        
        # 2. URL Pre-processing (M47)
        original_fact = fact
        if self._is_url(fact) and not preloaded_context:
            fetched_text = await self._resolve_url_content(fact)
            if fetched_text:
                # Restore LLM cleaning for UX quality (clean text in report)
                cleaned_article = await self.agent.clean_article(fetched_text)
                fact = cleaned_article or fetched_text
                final_context = fact

        # 3. Parallel Execution: Claims Extraction vs Oracle Check
        # Broadcast start of both
        if progress_callback: 
            # We refresh the status or keep it as analyzing_input (or move to extracting_claims early)
            # But let's signal we are moving to claim extraction specifically now.
            await progress_callback("extracting_claims")
        
        # Start Oracle Check immediately with raw fact (M54: Fast Path)
        # Fix M55: Use first significant sentence/chunk (~150 chars) instead of 300-char block
        # to improve Google Fact Check hit rate.
        fact_first_line = fact.strip().split('\n')[0]
        # Regex to find first sentence ending in ./!/? but handle abbreviations slightly?
        # Simple heuristic: Split by . and take first if lengthy enough, else take first line.
        blob = fact_first_line if len(fact_first_line) > 20 else fact[:200]
        # Further trim to ~150 chars at last space
        if len(blob) > 150:
             blob = blob[:150].rsplit(' ', 1)[0]
             
        fast_query = normalize_search_query(blob)
        task_oracle = asyncio.create_task(self.search_mgr.check_oracle(fast_query, content_lang or lang))
        
        # Start Claims Extraction (CPU/LLM intensive)
        cleaned_fact = clean_article_text(fact)
        task_claims = asyncio.create_task(self.agent.extract_claims(cleaned_fact[:4000], lang=lang))
        
        # Wait for Oracle first (it's faster usually)
        # We don't cancel claims yet, we wait to see if Oracle hits AND is verified.
        # But to be truly parallel, we await Oracle, then decide.
        
        oracle_res = await task_oracle
        
        if oracle_res:
             # Verify Semantic Relevance (M54: Critical Fix)
             # Extract the claim text from oracle result for comparison
             # Oracle res structure: see google_fact_check.py (has 'sources' list, 'rationale')
             # It doesn't strictly have a 'claim' field in the root usually (it's in sources[0]['title'] or similar).
             # But our tool returns a dict with 'rationale' which contains the claim.
             
             oracle_claim_text = oracle_res.get("rationale", "")
             oracle_rating = "Unknown" # Extracted in rationale mostly
             
             is_relevant = await self.agent.verify_oracle_relevance(original_fact, oracle_claim_text, oracle_rating)
             
             if is_relevant:
                 logger.info("[Pipeline] Oracle hit VERIFIED. Stopping.")
                 # Cancel claims extraction to save resources
                 task_claims.cancel()
                 try:
                     await task_claims
                 except asyncio.CancelledError:
                     pass
                 return self._finalize_oracle(oracle_res, original_fact, max_cost, include_internal)
             else:
                 logger.info("[Pipeline] Oracle hit REJECTED (irrelevant). Continuing to Waterfall.")

        # If we are here, Oracle missed or was irrelevant.
        # Wait for claims
        if progress_callback: await progress_callback("extracting_claims")
        try:
            claims = await task_claims
        except asyncio.CancelledError:
            # Should not happen unless we cancelled it
            claims = []

        # Generate Queries
        search_queries = self._generate_initial_queries(claims, fact)
        
        # 5. Parallel Waterfall Search (Tier 1 + Tier 2)
        if not preloaded_context:
            if progress_callback: await progress_callback("searching_web")
            
            search_lang = content_lang or lang
            tier1_domains = get_trusted_domains_by_lang(search_lang)
            
            # Select queries
            # T1 uses content-lang specific query if possible
            t1_query = search_queries[1] if len(search_queries) > 1 else search_queries[0]
            # T2 (General) uses broad query
            t2_query = search_queries[0]
            
            # Determine if we run Tier 2
            # For parallel execution, the logic "run T2 if T1 yields weak results" is hard purely in parallel
            # unless we speculate.
            # Speculation: If search_type is "standard", maybe just T1?
            # User wants "parallelization". 
            # Strategy: Run both if budget allows. OR run T1, then T2 (traditional waterfall).
            # The prompt asked for "parallelization". So let's run them parallel.
            # But we must ensure no duplicates.
            # T1: include_domains=TRUSTED
            # T2: exclude_domains=TRUSTED
            
            should_run_tier2 = (search_type in ["advanced", "deep"]) or (len(tier1_domains) < 5) # If few trusted domains, assume we need general search
            
            # Correction: Waterfall logic "Tier 1 -> if bad -> Tier 2" is better for cost/relevance usually.
            # But latency-wise, parallel is king.
            # Compromise: Always run T1. Run T2 parallel if "advanced" mode. 
            # If "standard", run T1, wait, check quality, then T2.
            
            tasks = []
            
            # Task T1
            if self._can_add_search(gpt_model, search_type, max_cost):
                tasks.append(self.search_mgr.search_tier1(t1_query, tier1_domains))
            
            # Task T2 (if applicable)
            # Parallel only if 'advanced' or explicitly requested. Default to waterfall for standard (to save cost/noise).
            run_t2_parallel = (search_type in ["advanced", "deep"])
            
            if run_t2_parallel and self._can_add_search(gpt_model, search_type, max_cost):
                tasks.append(self.search_mgr.search_tier2(t2_query, exclude_domains=tier1_domains))
            
            # Execute Parallel Tasks
            results = await asyncio.gather(*tasks)
            
            # Process T1
            if len(results) > 0:
                ctx1, srcs1 = results[0]
                final_context += "\n" + ctx1
                final_sources.extend(srcs1)
            
            # Process T2 (if ran parallel)
            if run_t2_parallel and len(results) > 1:
                ctx2, srcs2 = results[1]
                final_context += "\n\n=== GENERAL SEARCH ===\n" + ctx2
                final_sources.extend(srcs2)
            
            # Waterfall Fallback (if T2 didn't run parallel AND T1 was weak)
            if not run_t2_parallel and len(final_sources) < 2 and self._can_add_search(gpt_model, search_type, max_cost):
                if progress_callback: await progress_callback("searching_tier2_fallback")
                # Now we know T1 domains, but we can just exclude the predefined TRUSTED list to be safe/consistent
                ctx2, srcs2 = await self.search_mgr.search_tier2(t2_query, exclude_domains=tier1_domains)
                if srcs2:
                    final_context += "\n\n=== GENERAL SEARCH ===\n" + ctx2
                    final_sources.extend(srcs2)
                else:
                    # CSE Fallback (Tier 3)
                     if self.search_mgr.tavily_calls > 0:
                         if progress_callback: await progress_callback("searching_cse")
                         cse_results = await self.search_mgr.search_google_cse(t1_query, lang=lang)
                         for res in cse_results:
                                final_sources.append({
                                    "url": res.get("link"),
                                    "domain": get_registrable_domain(res.get("link")),
                                    "title": res.get("title"),
                                    "content": res.get("snippet", ""),
                                    "source_type": "general",
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
        # Use extracted article text for display
        # With format='text' from Tavily, `fact` contains plain text (no markdown)
        result["text"] = fact
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
        """Fetch URL content via Tavily Extract. Cleaning happens in claim extraction."""
        # from spectrue_core.utils.trace import Trace (already imported)
        
        try:
            raw_text = await self.search_mgr.fetch_url_content(url)
            if not raw_text or len(raw_text) < 50:
                return None
            
            logger.info("[Pipeline] URL resolved: %d chars", len(raw_text))
            Trace.event("pipeline.url_resolved", {"original": url, "chars": len(raw_text)})
            return raw_text
                
        except Exception as e:
            logger.warning("[Pipeline] Failed to resolve URL: %s", e)
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

