# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.verification.rag_verifier import get_rag_verifier
from spectrue_core.tools.search_tool import WebSearchTool, TRUSTED_DOMAINS
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.verification.trusted_sources import get_domains_for_language, get_trusted_domains_by_lang
from spectrue_core.config import SpectrueConfig
import asyncio
import re

RAG_ONLY_COST = 0
SEARCH_COSTS = {"basic": 80, "advanced": 160}
# M27: Updated model costs for GPT-5 family
MODEL_COSTS = {"gpt-5-nano": 5, "gpt-5-mini": 20, "gpt-5.1": 100}


class FactVerifierComposite:
    def __init__(self, config: SpectrueConfig):
        self.config = config
        self.rag = get_rag_verifier(config)
        self.web_search_tool = WebSearchTool(config)
        self.google_tool = GoogleFactCheckTool(config)
        self.agent = FactCheckerAgent(config)
        self.rag_confidence_threshold = 0.75
        self.time_sensitive_ttl = 300

    def _is_time_sensitive(self, fact: str, lang: str) -> bool:
        """
        Check if fact requires recent data using simple heuristics.
        M27: Replaced local LLM with keyword-based detection.
        """
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
        """
        M29: Enrich sources with trust indicators.
        Checks each source URL against TRUSTED_SOURCES registry.
        
        Returns sources with added fields:
        - is_trusted: bool (True if domain is in any category)
        - trust_category: str | None (category name if trusted)
        """
        from spectrue_core.verification.trusted_sources import TRUSTED_SOURCES
        from urllib.parse import urlparse
        
        # Build reverse lookup: domain -> category
        domain_to_category: dict[str, str] = {}
        for category, domains in TRUSTED_SOURCES.items():
            for domain in domains:
                domain_to_category[domain.lower()] = category
        
        enriched = []
        for source in sources_list:
            source_copy = dict(source)  # Don't mutate original
            
            # Extract domain from URL
            url = source.get("link") or source.get("url") or ""
            try:
                parsed = urlparse(url)
                host = (parsed.netloc or "").lower()
                # Remove www. prefix
                if host.startswith("www."):
                    host = host[4:]
                
                # Check if domain or parent domain is trusted
                category = domain_to_category.get(host)
                if not category:
                    # Check parent domain (e.g., news.bbc.co.uk -> bbc.co.uk)
                    parts = host.split(".")
                    for i in range(len(parts) - 1):
                        parent = ".".join(parts[i:])
                        if parent in domain_to_category:
                            category = domain_to_category[parent]
                            break
                
                if category:
                    print(f"[Trust] Matched trusted domain: {host} -> {category}")
                else:
                    # Debug print for missed major domains
                    if "reuters" in host or "bbc" in host:
                        print(f"[Trust] MISSED trusted domain: {host}")

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
        
        # M29: Enrich sources with trust indicators before adding to result
        enriched_sources = self._enrich_sources_with_trust(sources_list)
        analysis_result["sources"] = enriched_sources
        analysis_result["cost"] = cost
        analysis_result["text"] = fact # Add original text to result
        
        # Calculate RGBA for frontend visualization (Orthogonal Metrics)
        # R = Danger (Red) - unchanged
        r = self._clamp(analysis_result.get("danger_score", 0.0))
        
        # G = Veracity (Green) - Pure verified_score (Decoupled from style/context)
        g = self._clamp(analysis_result.get("verified_score", 0.5))
        
        # B = Honesty (Blue) - Combination of Context and Style
        # Represents the quality of presentation and context integrity
        context_score = self._clamp(analysis_result.get("context_score", 0.5))
        style_score = self._clamp(analysis_result.get("style_score", 0.5))
        b = (context_score + style_score) / 2.0
        
        # A = Confidence (Alpha) - unchanged
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
        content_lang: str = None  # M31: Content language from detection
    ):
        # === M20 SMART WATERFALL SEARCH STRATEGY ===
        # Step 1: Generate queries (GPT-5 Nano) - serves Oracle, Tier1, and Deep Dive
        # Step 2: Oracle (Google FC) with query[0] - FREE
        # Step 3: Tier 1 (Trusted Domains) with query[0] - 1 query
        # Step 4: Deep Dive (General + Local) with query[1,2] - +2 queries
        # Expected savings: 60-70% for Oracle/Tier1-confirmable facts
        
        # === GLOBAL CONTEXT OPTIMIZATION ===
        # If preloaded context is provided (from global search), use it directly.
        if preloaded_context:
            if progress_callback:
                await progress_callback("using_global_context")
            
            print(f"[Waterfall] Using preloaded global context ({len(preloaded_context)} chars). Skipping search.")
            
            # M27: No compression needed - GPT-5 has 400k context window
            # Just use the context directly (truncate if extremely large)
            context_to_use = preloaded_context[:100000]  # 100k chars max safety
            sources_to_use = preloaded_sources or []
            
            if progress_callback:
                await progress_callback("ai_analysis")
            
            model_cost = MODEL_COSTS.get(gpt_model, 20)
            # Search cost is 0 because we reused global context
            return await self._get_final_analysis(
                fact, context_to_use, sources_to_use[:10], gpt_model,
                model_cost, lang, analysis_mode
            )

        # === STEP 1: Generate 2-3 queries (M31: Content-Aware) ===
        if progress_callback:
            await progress_callback("generating_queries")
        
        SHORT_TEXT_THRESHOLD = 300
        search_queries = [fact, fact]  # Fallback (2 queries)
        
        try:
            if len(fact) < SHORT_TEXT_THRESHOLD:
                # M28: Short text - using direct translation strategy (TranslationService migration needed)
                print(f"[Waterfall] Short text detected ({len(fact)} chars). Using direct translation strategy.")
                # from spectrue_api.services.translation_service import get_translation_service
                # translation_service = get_translation_service()
                
                # Query 1: Original text (native)
                native_query = fact.strip()
                # Query 2: Translated to English (TODO: Implement translation later)
                en_query = fact # Placeholder
                # en_query = await translation_service.translate(fact, target_lang="en")
                
                search_queries = [en_query, native_query]
                print(f"[Waterfall] Generated 2 queries (direct): {search_queries}")
            else:
                # M31: Long text - use GPT-5 Nano for 2-3 targeted queries
                queries_list = await self.agent.generate_search_queries(
                    fact, context=context_text, lang=lang, content_lang=content_lang  # M31: Pass content_lang
                )
                if queries_list and len(queries_list) > 0:
                    search_queries = queries_list  # Use all queries (2-3)
                    print(f"[Waterfall] Generated {len(queries_list)} queries (LLM): {search_queries}")
                else:
                    print("[Waterfall] GPT-5 Nano returned empty, using fallback.")
        except Exception as e:
            print(f"[Waterfall] Failed to generate queries: {e}. Using fallback.")
        
        # === STEP 2 & 2.5: Oracle (Google FC) and RAG (Local) in PARALLEL ===
        # Run independent checks concurrently to reduce latency
        if progress_callback:
            await progress_callback("checking_oracle")
            
        oracle_query = search_queries[0]
        print(f"[Waterfall] Oracle query: '{oracle_query[:100]}...'")
        
        # Define tasks
        async def check_oracle():
            return await self.google_tool.search(oracle_query, lang)
            
        async def check_rag():
            if self.rag.is_ready(lang):
                print(f"[Waterfall] Checking local RAG for lang '{lang}'...")
                try:
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        None, 
                        lambda: self.rag.verify(fact, lang, top_k=3)
                    )
                except Exception as e:
                    print(f"[Waterfall] RAG search failed: {e}")
            return None

        # Execute concurrently
        oracle_result, rag_result = await asyncio.gather(check_oracle(), check_rag())
        
        # Process Oracle Result
        if oracle_result:
            oracle_result["text"] = fact
            print("[Waterfall] ✓ Oracle hit (Google Fact Check). Stopping.")
            return oracle_result

        # Process RAG Result
        rag_sources = []
        rag_context = ""
        if rag_result and rag_result.get("relevance", 0) > 0.5:
            rag_sources = rag_result.get("sources", [])
            print(f"[Waterfall] ✓ RAG found {len(rag_sources)} sources (relevance: {rag_result['relevance']:.2f})")
            rag_context = "\n".join([f"Source: {s['title']}\n{s['snippet']}" for s in rag_sources])
            rag_context = f"=== LOCAL KNOWLEDGE BASE (RAG) ===\n{rag_context}\n"
        else:
             if rag_result:
                 print(f"[Waterfall] RAG relevance too low ({rag_result.get('relevance', 0):.2f}). Ignoring.")

        # === STEP 3: Search Tier 1 (Trusted Domains) ===
        ttl = self.time_sensitive_ttl if self._is_time_sensitive(fact, lang) else None
        
        if progress_callback:
            await progress_callback("searching_tier1")
        
        # === STEP 3: Tier 1 (Trusted Domains) with query[0] ===
        if progress_callback:
            await progress_callback("searching_tier1")
        
        # M31: Use content_lang for trusted domains if available
        search_lang = content_lang if content_lang else lang
        tier1_domains = get_trusted_domains_by_lang(search_lang)
        print(f"[Waterfall] Tier 1 domains for lang='{search_lang}': {len(tier1_domains)} domains")
        
        # M31: Use content language query if available (query[1]), else English query (query[0])
        tier1_query = search_queries[1] if len(search_queries) > 1 and content_lang else search_queries[0]
        print(f"[Waterfall] Tier 1 query: '{tier1_query[:80]}...'")
        
        tier1_context, tier1_sources = await self.web_search_tool.search(
            tier1_query,  # Use content language query
            search_depth=search_type,
            ttl=ttl,
            domains=tier1_domains
        )
        
        # Merge RAG results into Tier 1
        if rag_sources:
            tier1_sources.extend(rag_sources)
            tier1_context = f"{rag_context}\n=== WEB SEARCH (TIER 1) ===\n{tier1_context}"
        
        # === DECISION: If Tier 1 found enough results, stop here (save 2 queries) ===
        # M20: Require at least 3 sources to be confident enough to skip general search.
        if len(tier1_sources) >= 3:
            print(f"[Waterfall] ✓ Tier 1 strong match ({len(tier1_sources)} sources). Skipping general search.")
            
            # M27: No compression needed - GPT-5 has 400k context window
            if progress_callback:
                await progress_callback("ai_analysis")
            
            search_cost = SEARCH_COSTS.get(search_type, 80)
            model_cost = MODEL_COSTS.get(gpt_model, 20)
            return await self._get_final_analysis(
                fact, tier1_context, tier1_sources[:10], gpt_model,
                search_cost + model_cost, lang, analysis_mode
            )

        # === STEP 4: Deep Dive (EN + Native) ===
        # M28: Now using only 2 queries
        print("[Waterfall] Tier 1 silent. Running deep dive (EN + Native)...")
        
        if progress_callback:
            await progress_callback("searching_deep")
        
        # Run both searches concurrently with our 2 queries
        en_task = self.web_search_tool.search(
            search_queries[0],  # English query
            search_depth=search_type,
            ttl=ttl
        )
        native_task = self.web_search_tool.search(
            search_queries[1],  # Native language query
            search_depth=search_type,
            ttl=ttl
        )
        
        (en_context, en_sources), (native_context, native_sources) = await asyncio.gather(
            en_task, native_task
        )
        
        # === AGGREGATION: Combine all sources ===
        all_sources = tier1_sources + en_sources + native_sources
        all_context = f"{tier1_context}\n{en_context}\n{native_context}"
        
        # Deduplicate sources by URL
        seen_urls = set()
        unique_sources = []
        for src in all_sources:
            url = src.get("link", "")
            if url and url not in seen_urls:
                unique_sources.append(src)
                seen_urls.add(url)
        
        print(f"[Waterfall] Deep dive complete: {len(unique_sources)} unique sources (from {len(all_sources)} total)")
        
        # M27: No compression - GPT-5 has 400k context
        if len(all_context) > 100000:
            print(f"[M27] Context too large ({len(all_context)} chars), truncating to 100k")
            all_context = all_context[:100000]
        
        # M28: Cost = 2 deep searches + Tier 1 search + model
        if progress_callback:
            await progress_callback("ai_analysis")
        search_cost = SEARCH_COSTS.get(search_type, 80) * 3  # Tier1 + EN + Native
        model_cost = MODEL_COSTS.get(gpt_model, 20)
        total_cost = search_cost + model_cost
        
        return await self._get_final_analysis(
            fact, all_context, unique_sources[:10], gpt_model,
            total_cost, lang, analysis_mode
        )
