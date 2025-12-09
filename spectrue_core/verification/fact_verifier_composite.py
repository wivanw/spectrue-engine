from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.tools.search_tool import WebSearchTool
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.verification.trusted_sources import get_trusted_domains_by_lang
from spectrue_core.config import SpectrueConfig
import asyncio

RAG_ONLY_COST = 0
SEARCH_COSTS = {"basic": 80, "advanced": 160}
MODEL_COSTS = {"gpt-5-nano": 5, "gpt-5-mini": 20, "gpt-5.1": 100}


class FactVerifierComposite:
    def __init__(self, config: SpectrueConfig):
        self.config = config
        self.web_search_tool = WebSearchTool(config)
        self.google_tool = GoogleFactCheckTool(config)
        self.agent = FactCheckerAgent(config)
        self.time_sensitive_ttl = 300

    def _is_time_sensitive(self, fact: str, lang: str) -> bool:
        """Check if fact requires recent data using simple heuristics."""
        time_keywords = [
            "today", "yesterday", "this week", "this month", "breaking",
            "just announced", "latest", "new", "recent", "now",
            "сьогодні", "вчора", "щойно", "новий", "останній",
            "сегодня", "вчера", "только что", "новый", "последний"
        ]
        fact_lower = fact.lower()
        return any(kw in fact_lower for kw in time_keywords)

    def _clamp(self, value) -> float:
        """Clamp value between 0.0 and 1.0."""
        try:
            if value is None:
                return 0.5
            val = float(value)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.5

    def _enrich_sources_with_trust(self, sources_list: list) -> list:
        """Enrich sources with trust indicators."""
        from spectrue_core.verification.trusted_sources import TRUSTED_SOURCES
        from urllib.parse import urlparse
        
        domain_to_category: dict[str, str] = {}
        for category, domains in TRUSTED_SOURCES.items():
            for domain in domains:
                domain_to_category[domain.lower()] = category
        
        enriched = []
        for source in sources_list:
            source_copy = dict(source)
            
            url = source.get("link") or source.get("url") or ""
            try:
                parsed = urlparse(url)
                host = (parsed.netloc or "").lower()
                if host.startswith("www."):
                    host = host[4:]
                
                category = domain_to_category.get(host)
                if not category:
                    parts = host.split(".")
                    for i in range(len(parts) - 1):
                        parent = ".".join(parts[i:])
                        if parent in domain_to_category:
                            category = domain_to_category[parent]
                            break
                
                source_copy["is_trusted"] = category is not None
                source_copy["trust_category"] = category
                
            except Exception:
                source_copy["is_trusted"] = False
                source_copy["trust_category"] = None
            
            enriched.append(source_copy)
        
        return enriched

    async def _get_final_analysis(self, fact: str, context: str, sources_list: list, gpt_model: str, cost: int,
                                  lang: str, analysis_mode: str = "general") -> dict:
        analysis_result = await self.agent.analyze(fact, context, gpt_model, lang, analysis_mode)
        
        enriched_sources = self._enrich_sources_with_trust(sources_list)
        analysis_result["sources"] = enriched_sources
        analysis_result["cost"] = cost
        analysis_result["text"] = fact
        
        r = self._clamp(analysis_result.get("danger_score", 0.0))
        g = self._clamp(analysis_result.get("verified_score", 0.5))
        context_score = self._clamp(analysis_result.get("context_score", 0.5))
        style_score = self._clamp(analysis_result.get("style_score", 0.5))
        b = (context_score + style_score) / 2.0
        a = self._clamp(analysis_result.get("confidence_score", 0.5))
        
        analysis_result["rgba"] = [r, g, b, a]
        
        return analysis_result

    async def verify_fact(
        self, 
        fact: str, 
        search_type: str, 
        gpt_model: str, 
        lang: str, 
        analysis_mode: str = "general", 
        progress_callback=None, 
        context_text: str = "", 
        preloaded_context: str = None, 
        preloaded_sources: list = None,
        content_lang: str = None
    ):
        """
        Web-only verification (no RAG).
        Strategy: Oracle -> Tier 1 -> Deep Dive
        """
        
        # Global context optimization
        if preloaded_context:
            if progress_callback:
                await progress_callback("using_global_context")
            
            print(f"[Waterfall] Using preloaded global context ({len(preloaded_context)} chars).")
            context_to_use = preloaded_context[:100000]
            sources_to_use = preloaded_sources or []
            
            if progress_callback:
                await progress_callback("ai_analysis")
            
            model_cost = MODEL_COSTS.get(gpt_model, 20)
            return await self._get_final_analysis(
                fact, context_to_use, sources_to_use[:10], gpt_model,
                model_cost, lang, analysis_mode
            )

        # Generate search queries
        if progress_callback:
            await progress_callback("generating_queries")
        
        SHORT_TEXT_THRESHOLD = 300
        search_queries = [fact, fact]
        
        try:
            if len(fact) < SHORT_TEXT_THRESHOLD:
                print(f"[Waterfall] Short text ({len(fact)} chars). Using direct strategy.")
                native_query = fact.strip()
                en_query = fact
                search_queries = [en_query, native_query]
            else:
                queries_list = await self.agent.generate_search_queries(
                    fact, context=context_text, lang=lang, content_lang=content_lang
                )
                if queries_list and len(queries_list) > 0:
                    search_queries = queries_list
                    print(f"[Waterfall] Generated {len(queries_list)} queries (LLM): {search_queries}")
                else:
                    print("[Waterfall] GPT-5 Nano returned empty, using fallback.")
        except Exception as e:
            print(f"[Waterfall] Failed to generate queries: {e}. Using fallback.")
        
        # Oracle (Google Fact Check)
        if progress_callback:
            await progress_callback("checking_oracle")
            
        oracle_query = search_queries[0]
        print(f"[Waterfall] Oracle query: '{oracle_query[:100]}...'")
        
        oracle_result = await self.google_tool.search(oracle_query, lang)
        
        if oracle_result:
            oracle_result["text"] = fact
            print("[Waterfall] ✓ Oracle hit (Google Fact Check). Stopping.")
            return oracle_result

        # Tier 1 (Trusted Domains)
        ttl = self.time_sensitive_ttl if self._is_time_sensitive(fact, lang) else None
        
        if progress_callback:
            await progress_callback("searching_tier1")
        
        search_lang = content_lang if content_lang else lang
        tier1_domains = get_trusted_domains_by_lang(search_lang)
        print(f"[Waterfall] Tier 1 domains for lang='{search_lang}': {len(tier1_domains)} domains")
        
        tier1_query = search_queries[1] if len(search_queries) > 1 and content_lang else search_queries[0]
        print(f"[Waterfall] Tier 1 query: '{tier1_query[:80]}...'")
        
        tier1_context, tier1_sources = await self.web_search_tool.search(
            tier1_query,
            search_depth=search_type,
            ttl=ttl,
            domains=tier1_domains
        )
        
        if len(tier1_sources) >= 3:
            print(f"[Waterfall] ✓ Tier 1 strong match ({len(tier1_sources)} sources). Skipping general search.")
            
            if progress_callback:
                await progress_callback("ai_analysis")
            
            search_cost = SEARCH_COSTS.get(search_type, 80)
            model_cost = MODEL_COSTS.get(gpt_model, 20)
            return await self._get_final_analysis(
                fact, tier1_context, tier1_sources[:10], gpt_model,
                search_cost + model_cost, lang, analysis_mode
            )

        # Deep Dive (EN + Native)
        print("[Waterfall] Tier 1 silent. Running deep dive (EN + Native)...")
        
        if progress_callback:
            await progress_callback("searching_deep")
        
        en_task = self.web_search_tool.search(
            search_queries[0],
            search_depth=search_type,
            ttl=ttl
        )
        native_task = self.web_search_tool.search(
            search_queries[1],
            search_depth=search_type,
            ttl=ttl
        )
        
        (en_context, en_sources), (native_context, native_sources) = await asyncio.gather(
            en_task, native_task
        )
        
        # Aggregation
        all_sources = tier1_sources + en_sources + native_sources
        all_context = f"{tier1_context}\n{en_context}\n{native_context}"
        
        seen_urls = set()
        unique_sources = []
        for src in all_sources:
            url = src.get("link", "")
            if url and url not in seen_urls:
                unique_sources.append(src)
                seen_urls.add(url)
        
        print(f"[Waterfall] Deep dive complete: {len(unique_sources)} unique sources")
        
        if len(all_context) > 100000:
            all_context = all_context[:100000]
        
        if progress_callback:
            await progress_callback("ai_analysis")
        search_cost = SEARCH_COSTS.get(search_type, 80) * 3
        model_cost = MODEL_COSTS.get(gpt_model, 20)
        total_cost = search_cost + model_cost
        
        return await self._get_final_analysis(
            fact, all_context, unique_sources[:10], gpt_model,
            total_cost, lang, analysis_mode
        )
