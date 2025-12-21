from spectrue_core.verification.search_mgr import SearchManager, SEARCH_COSTS
from spectrue_core.verification.evidence import build_evidence_pack
from spectrue_core.verification.trusted_sources import is_social_platform
from spectrue_core.utils.text_processing import clean_article_text, normalize_search_query
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.utils.trust_utils import enrich_sources_with_trust
from spectrue_core.utils.trace import Trace
from spectrue_core.config import SpectrueConfig
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.agents.skills.oracle_validation import EVIDENCE_THRESHOLD
from spectrue_core.graph import ClaimGraphBuilder
import logging
import asyncio
import re

logger = logging.getLogger(__name__)

# M63: Intents that should trigger Oracle check
ORACLE_CHECK_INTENTS = {"news", "evergreen", "official"}
# M63: Intents that should skip Oracle (opinion, prediction)
ORACLE_SKIP_INTENTS = {"opinion", "prediction"}


class ValidationPipeline:
    """
    Orchestrates the fact-checking waterfall process.
    """
    def __init__(self, config: SpectrueConfig, agent: FactCheckerAgent, translation_service=None):
        self.config = config
        self.agent = agent
        # M63: Pass oracle_skill to SearchManager for hybrid mode
        self.search_mgr = SearchManager(config, oracle_validator=agent.oracle_skill)
        # M67: Optional translation service for Oracle result localization
        self.translation_service = translation_service
        
        # M72: ClaimGraph for key claim identification
        self._claim_graph: ClaimGraphBuilder | None = None
        if config and config.runtime.claim_graph.enabled:
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(api_key=config.openai_api_key)
            self._claim_graph = ClaimGraphBuilder(
                config=config.runtime.claim_graph,
                openai_client=openai_client,
                edge_typing_skill=agent.edge_typing_skill,
            )

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
        needs_cleaning: bool = False,  # M60: Text from extension needs LLM cleaning
        source_url: str | None = None,  # M60: Original URL for inline source exclusion
    ) -> dict:
        
        # 1. Initialize
        if progress_callback:
            await progress_callback("analyzing_input")
        
        self.search_mgr.reset_metrics()
        final_context = preloaded_context or ""
        final_sources = preloaded_sources or []
        
        # 2. URL Pre-processing (M47 + M60 improved)
        original_fact = fact
        inline_sources = []  # M60: URLs extracted from article text
        exclude_url = source_url  # For inline source exclusion
        
        # Case A: URL provided -> fetch via Tavily + clean
        if self._is_url(fact) and not preloaded_context:
            exclude_url = fact
            fetched_text = await self._resolve_url_content(fact)
            if fetched_text:
                # M60: Extract URL-anchor pairs BEFORE cleaning
                url_anchors = self._extract_url_anchors(fetched_text, exclude_url=exclude_url)
                if url_anchors:
                    logger.info("[Pipeline] Found %d URL-anchor pairs in raw text", len(url_anchors))
                
                # M73.5: Warn user about large text processing
                if len(fetched_text) > 10000 and progress_callback:
                    await progress_callback("processing_large_text")
                    logger.info("[Pipeline] Large text detected: %d chars, extended timeout", len(fetched_text))
                
                # LLM cleaning for UX quality
                cleaned_article = await self.agent.clean_article(fetched_text)
                fact = cleaned_article or fetched_text
                final_context = fact
                
                # M60: Restore URLs for anchors that survived cleaning
                # T7: Store as candidates, don't add to final_sources yet - will verify after claims extraction
                if url_anchors and cleaned_article:
                    inline_sources = self._restore_urls_from_anchors(cleaned_article, url_anchors)
                    if inline_sources:
                        logger.info("[Pipeline] Restored %d inline source candidates after cleaning", len(inline_sources))
                        Trace.event("pipeline.inline_sources", {"count": len(inline_sources), "urls": [s["url"][:80] for s in inline_sources[:5]]})
                        # T7: Mark as candidates, will be verified after claims extraction
                        for src in inline_sources:
                            src["is_primary_candidate"] = True
        
        # Case B: Text from extension (full page) needs cleaning (no Tavily fetch needed)
        elif needs_cleaning and not self._is_url(fact):
            logger.info("[Pipeline] Extension page mode: cleaning %d chars", len(fact))
            
            # M60: Extract URL-anchor pairs BEFORE cleaning
            url_anchors = self._extract_url_anchors(fact, exclude_url=exclude_url)
            if url_anchors:
                logger.info("[Pipeline] Found %d URL-anchor pairs in extension text", len(url_anchors))
            
            # M73.5: Warn user about large text processing
            if len(fact) > 10000 and progress_callback:
                await progress_callback("processing_large_text")
                logger.info("[Pipeline] Large text detected: %d chars, extended timeout", len(fact))
            
            # LLM cleaning
            cleaned_article = await self.agent.clean_article(fact)
            if cleaned_article:
                # M60: Restore URLs for anchors that survived cleaning
                # T7: Store as candidates, don't add to final_sources yet
                if url_anchors:
                    inline_sources = self._restore_urls_from_anchors(cleaned_article, url_anchors)
                    if inline_sources:
                        logger.info("[Pipeline] Restored %d inline source candidates after cleaning", len(inline_sources))
                        Trace.event("pipeline.inline_sources", {"count": len(inline_sources), "urls": [s["url"][:80] for s in inline_sources[:5]]})
                        for src in inline_sources:
                            src["is_primary_candidate"] = True
                
                fact = cleaned_article
                final_context = fact
        
        # Case C: Plain text (no cleaning needed) - just extract inline URLs
        elif not self._is_url(fact) and not needs_cleaning:
            # Still extract URLs from text (user pasted text with links)
            url_anchors = self._extract_url_anchors(fact, exclude_url=exclude_url)
            if url_anchors:
                logger.info("[Pipeline] Found %d URL-anchor pairs in plain text", len(url_anchors))
                # No cleaning needed, so all anchors survive - T7: mark as candidates
                for item in url_anchors:
                    inline_sources.append({
                        "url": item["url"],
                        "title": item["anchor"],
                        "domain": item["domain"],
                        "source_type": "inline",
                        "is_trusted": False,
                        "is_primary_candidate": True  # T7: Will verify after claims
                    })
                if inline_sources:
                    Trace.event("pipeline.inline_sources", {"count": len(inline_sources), "urls": [s["url"][:80] for s in inline_sources[:5]]})

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
        # Start Claims Extraction (CPU/LLM intensive)
        cleaned_fact = clean_article_text(fact)
        task_claims = asyncio.create_task(self.agent.extract_claims(cleaned_fact[:4000], lang=lang))
        
        # Determine fast query for potential Oracle use
        if len(blob) > 150:
             blob = blob[:150].rsplit(' ', 1)[0]
        fast_query = normalize_search_query(blob)

        # Wait for claims to determine if we need Oracle
        if progress_callback:
            await progress_callback("extracting_claims")
        try:
            claims_result = await task_claims
            # M63: Now returns 3-tuple: (claims, should_check_oracle, article_intent)
            if isinstance(claims_result, tuple) and len(claims_result) >= 3:
                claims, should_check_oracle, article_intent = claims_result
            elif isinstance(claims_result, tuple) and len(claims_result) == 2:
                # Backwards compatibility
                claims, should_check_oracle = claims_result
                article_intent = "news"  # Default
            else:
                claims, should_check_oracle, article_intent = claims_result, False, "news"
        except asyncio.CancelledError:
            claims = []
            should_check_oracle = False
            article_intent = "news"
        
        # M63: Assign IDs to claims EARLY (needed for Oracle mapping)
        if claims:
            for i, c in enumerate(claims):
                c["id"] = f"c{i+1}"
        
        # T7: Verify inline source relevance against extracted claims
        if inline_sources and claims:
            if progress_callback:
                await progress_callback("verifying_sources")
            
            article_excerpt = fact[:500] if fact else ""
            verified_inline_sources = []
            
            # Run verification for each inline source (parallel for speed)
            verification_tasks = [
                self.agent.verify_inline_source_relevance(claims, src, article_excerpt)
                for src in inline_sources
            ]
            verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            for src, result in zip(inline_sources, verification_results):
                if isinstance(result, Exception):
                    logger.warning("[Pipeline] Inline source verification failed: %s", result)
                    # On error, include as secondary source
                    src["is_primary"] = False
                    src["is_relevant"] = True
                    verified_inline_sources.append(src)
                else:
                    is_relevant = result.get("is_relevant", True)
                    is_primary = result.get("is_primary", False)
                    
                    if is_relevant:
                        src["is_primary"] = is_primary
                        src["is_relevant"] = True
                        # T7: Primary sources should be protected from deduplication
                        if is_primary:
                            src["is_trusted"] = True  # Treat primary as trusted for scoring weight
                        verified_inline_sources.append(src)
                        logger.info("[Pipeline] Inline source %s: relevant=%s, primary=%s", 
                                   src.get("domain"), is_relevant, is_primary)
                    else:
                        logger.debug("[Pipeline] Inline source rejected: %s", src.get("domain"))
            
            if verified_inline_sources:
                Trace.event("pipeline.inline_sources_verified", {
                    "total": len(inline_sources),
                    "passed": len(verified_inline_sources),
                    "primary": len([s for s in verified_inline_sources if s.get("is_primary")])
                })
                final_sources.extend(verified_inline_sources)
        elif inline_sources:
            # No claims extracted - add inline sources as-is (best effort)
            logger.info("[Pipeline] No claims extracted, adding %d inline sources as secondary", len(inline_sources))
            for src in inline_sources:
                src["is_primary"] = False
                src["is_relevant"] = True
            final_sources.extend(inline_sources)
        
        # M63: Hybrid Oracle Flow
        # Check Oracle based on article_intent (news/evergreen/official) OR should_check_oracle flag
        run_oracle = (
            (article_intent in ORACLE_CHECK_INTENTS) or 
            should_check_oracle
        ) and (article_intent not in ORACLE_SKIP_INTENTS)
        
        oracle_evidence_source = None  # Will hold Oracle source if EVIDENCE scenario
        
        if run_oracle:
            if progress_callback:
                await progress_callback("checking_oracle")
            
            # M63: Identify claims to check - use normalized_text if available
            candidates = [c for c in claims if c.get("check_oracle")]
            
            # Fallback: Use first core claim with high importance
            if not candidates:
                core_claims = [c for c in claims if c.get("type") == "core"]
                if core_claims:
                    # Use normalized_text for better Oracle matching
                    candidates = [core_claims[0]]
                else:
                    # Use fast_query as last resort
                    candidates = [{"text": fast_query, "normalized_text": fast_query}]
            
            # Limit to 1 candidate to preserve quota (more targeted now)
            candidates = candidates[:1]
            
            for cand in candidates:
                # M71: Try English query first (Google Fact Check has better EN coverage)
                # Look for English query in query_candidates
                en_query = None
                for qc in cand.get("query_candidates", []):
                    qc_text = qc.get("text", "")
                    # Simple heuristic: English if mostly ASCII
                    if qc_text and sum(1 for c in qc_text if ord(c) < 128) / len(qc_text) > 0.9:
                        en_query = qc_text
                        break
                
                # Prefer English query, fallback to normalized_text
                if en_query:
                    query_text = en_query.strip(" ,.-:;")
                    logger.info("[Pipeline] Oracle: Using English query candidate")
                else:
                    query_text = (cand.get("normalized_text") or cand.get("text", "")).strip(" ,.-:;")
                
                q = normalize_search_query(query_text)
                
                logger.info("[Pipeline] Oracle Hybrid Check: intent=%s, query=%s", article_intent, q[:300])
                
                # M63: Use new hybrid method with LLM validation
                oracle_result = await self.search_mgr.check_oracle_hybrid(q, intent=article_intent)
                
                if not oracle_result:
                    continue

                status = oracle_result.get("status", "EMPTY")
                relevance = oracle_result.get("relevance_score", 0.0)
                is_jackpot = oracle_result.get("is_jackpot", False)
                
                # M73.4: Build trace payload with error details if applicable
                trace_payload = {
                    "intent": article_intent,
                    "query_used": q[:300],
                    "relevance_score": relevance,
                    "status": status,
                    "is_jackpot": is_jackpot
                }
                
                # M73.4: Handle ERROR status (API failure)
                if status == "ERROR":
                    code = oracle_result.get("error_status_code")
                    detail = oracle_result.get("error_detail", "")
                    trace_payload.update({"error_code": code, "error_detail": detail})
                    Trace.event("pipeline.oracle_hybrid_error", trace_payload)
                    
                    logger.warning("[Pipeline] Oracle API ERROR (%s): %s", code, detail)
                    
                    # Break loop on auth/quota errors, continue on timeouts
                    if code in (403, 429):
                        logger.error("[Pipeline] Oracle Quota Exceeded/Auth Error. Stopping Oracle loop.")
                        break 
                    continue
                
                # M73.4: Handle DISABLED status (no validator configured)
                if status == "DISABLED":
                    Trace.event("pipeline.oracle_hybrid", {**trace_payload, "disabled": True})
                    logger.warning("[Pipeline] Oracle DISABLED (no LLM validator). Skipping.")
                    break
                
                Trace.event("pipeline.oracle_hybrid", trace_payload)
                
                if status == "EMPTY":
                    logger.info("[Pipeline] Oracle: No results found. Continuing to search.")
                    continue
                
                # SCENARIO A: JACKPOT (relevance > 0.9)
                if is_jackpot:
                    logger.info(
                        "[Pipeline] 🎰 JACKPOT! Oracle hit (score=%.2f). Stopping pipeline.",
                        relevance
                    )
                    oracle_final = await self._finalize_oracle_hybrid(
                        oracle_result, 
                        original_fact, 
                        lang=lang, 
                        progress_callback=progress_callback
                    )
                    Trace.event("pipeline.result", {
                        "type": "oracle_jackpot",
                        "verified_score": oracle_final.get("verified_score"),
                        "status": status,
                        "source": oracle_result.get("publisher"),
                    })
                    return oracle_final
                
                # SCENARIO B: EVIDENCE (0.5 < relevance <= 0.9)
                elif relevance > EVIDENCE_THRESHOLD:
                    logger.info(
                        "[Pipeline] 📚 EVIDENCE: Oracle related (score=%.2f). Adding to pack.",
                        relevance
                    )
                    # Create source from Oracle result for evidence pack
                    oracle_evidence_source = self._create_oracle_source(oracle_result)
                    # Bind Oracle evidence to the specific claim it verifies
                    oracle_evidence_source["claim_id"] = cand.get("id")
                    # Don't stop - continue to Tavily for more context
                
                # SCENARIO C: MISS (relevance <= 0.5)
                else:
                    logger.info(
                        "[Pipeline] Oracle MISS (score=%.2f). Proceeding to search.",
                        relevance
                    )
            
            if not oracle_evidence_source:
                logger.info("[Pipeline] Oracle checks finished. No relevant hits.")
        else:
            skip_reason = "opinion/prediction intent" if article_intent in ORACLE_SKIP_INTENTS else "no markers"
            logger.info("[Pipeline] Skipping Oracle (%s, intent=%s)", skip_reason, article_intent)
        
        # Add Oracle evidence source if we got one (EVIDENCE scenario)
        if oracle_evidence_source:
            final_sources.append(oracle_evidence_source)

        # M72: ClaimGraph - identify key claims for query prioritization
        key_claim_ids: set[str] = set()
        graph_result = None  # M73: Store for claim enrichment
        if self._claim_graph and claims:
            if progress_callback:
                await progress_callback("building_claim_graph")
            
            try:
                graph_result = await self._claim_graph.build(claims)
                
                if not graph_result.disabled:
                    key_claim_ids = set(graph_result.key_claim_ids)
                    # M72: Basic key claim boost
                    for claim in claims:
                        if claim.get("id") in key_claim_ids:
                            claim["importance"] = min(1.0, claim.get("importance", 0.5) + 0.2)
                
                # Always trace results
                Trace.event("claim_graph", graph_result.to_trace_dict())
                
                if graph_result.disabled:
                    logger.info("[M72] ClaimGraph disabled: %s", graph_result.disabled_reason)
                else:
                    logger.info("[M72] ClaimGraph: %d key claims identified", len(key_claim_ids))
                    
            except Exception as e:
                logger.warning("[M72] ClaimGraph failed: %s. Fallback to original flow.", e)
                Trace.event("claim_graph", {"enabled": True, "error": str(e)[:100]})
        
        # M73 Layer 2-3: Claim Enrichment with Graph Signals
        enriched_count = 0
        high_tension_count = 0
        if graph_result and not graph_result.disabled:
            cfg = self.config.runtime.claim_graph
            
            # Layer 2: Structural Prioritization
            if cfg.structural_prioritization_enabled:
                for claim in claims:
                    claim_id = claim.get("id")
                    if not claim_id:
                        continue
                    
                    ranked = graph_result.get_ranked_by_id(claim_id)
                    if ranked:
                        # Enrich claim with graph metrics
                        claim["graph_centrality"] = ranked.centrality_score
                        claim["graph_structural_weight"] = ranked.in_structural_weight
                        claim["graph_tension_score"] = ranked.in_contradict_weight
                        claim["is_key_claim"] = ranked.is_key_claim
                        enriched_count += 1
                        
                        # M73 Layer 2: Structural weight boost
                        if ranked.in_structural_weight > cfg.structural_weight_threshold:
                            claim["importance"] = min(1.0, claim.get("importance", 0.5) + cfg.structural_boost)
                        
                        # M73 Layer 3: Tension boost (high contradiction = needs verification)
                        if cfg.tension_signal_enabled and ranked.in_contradict_weight > cfg.tension_threshold:
                            claim["importance"] = min(1.0, claim.get("importance", 0.5) + cfg.tension_boost)
                            high_tension_count += 1
                
                # Trace enrichment results
                Trace.event("claim_intelligence", {
                    "structural_prioritization_enabled": True,
                    "tension_signal_enabled": cfg.tension_signal_enabled,
                    "claims_enriched": enriched_count,
                    "high_tension_claims": high_tension_count,
                    "key_claims_with_scores": [
                        {
                            "id": c.claim_id,
                            "centrality": round(c.centrality_score, 4),
                            "structural": round(c.in_structural_weight, 2),
                            "tension": round(c.in_contradict_weight, 2),
                        }
                        for c in graph_result.key_claims[:5]
                    ],
                })
                
                if enriched_count > 0:
                    logger.info("[M73] Enriched %d claims with graph signals (%d high-tension)", 
                               enriched_count, high_tension_count)
        
        # M73 Layer 4: Evidence-Need Routing Tracing
        if self.config.runtime.claim_graph.evidence_need_routing_enabled and claims:
            evidence_need_dist: dict[str, int] = {}
            for claim in claims:
                need = claim.get("evidence_need", "unknown")
                evidence_need_dist[need] = evidence_need_dist.get(need, 0) + 1
            
            Trace.event("evidence_need_routing", {
                "enabled": True,
                "distribution": evidence_need_dist,
                "sample": [
                    {"id": c.get("id"), "evidence_need": c.get("evidence_need", "unknown")}
                    for c in claims[:3]
                ],
            })


        # M62: Smart Query Selection (typed priority slots)
        search_queries = self._select_diverse_queries(claims, max_queries=3, fact_fallback=fact)
        
        # 5. M65: Unified Search (cost-optimized waterfall)
        if not preloaded_context:
            if progress_callback:
                await progress_callback("searching_unified")
            
            
            # Determine Tavily topic based on article intent
            # M66: Smart Routing - Use claim search_method if available
            tavily_topic = "general"
            
            # Check if any significant claim requests "news" method
            claims_need_news = any(
                c.get("search_method") == "news" 
                for c in claims 
                if c.get("importance", 0) >= 0.5
            )
            
            if claims_need_news or article_intent in ("news", "opinion"):
                tavily_topic = "news"
            
            # M65: Use primary query for single unified search
            # This replaces the wasteful Tier 1 -> Tier 2 (same query) pattern.
            primary_query = search_queries[0] if search_queries else ""
            
            has_results = False
            
            if primary_query and self._can_add_search(gpt_model, search_type, max_cost):
                current_topic = tavily_topic
                
                # M67: Semantic Refinement Loop (Max 2 Attempts)
                for attempt in range(2):
                    logger.info("[M65/M67] Unified Search (Attempt %d): %s (topic=%s)", attempt+1, primary_query[:50], current_topic)
                    
                    u_ctx, u_srcs = await self.search_mgr.search_unified(
                        primary_query, 
                        topic=current_topic,
                        intent=article_intent
                    )
                    
                    if not u_srcs:
                         # If no results, break to allow fallback to CSE
                         break
                         
                    # Verify Relevance
                    gate_result = await self.agent.verify_search_relevance(claims, u_srcs)
                    gate_status = gate_result.get("status", "RELEVANT")
                    
                    if gate_status == "RELEVANT":
                        # Success
                        final_context += "\n" + u_ctx
                        final_sources.extend(u_srcs)
                        has_results = True
                        break
                    else:
                        logger.warning("[M67] Semantic Router: REJECTED results (%s). Reason: %s", gate_status, gate_result.get("reason"))
                        # Refinement: Switch topic for next attempt
                        if attempt == 0:
                            new_topic = "general" if current_topic == "news" else "news"
                            logger.info("[M67] Refining Search: Switching topic %s -> %s", current_topic, new_topic)
                            current_topic = new_topic
                            continue
                        else:
                             # Final failure after retry
                             return {
                                "verified_score": 0.0,
                                "confidence_score": 0.0,
                                "analysis": f"Search results rejected by Semantic Router ({gate_status}) after refinement. Reason: {gate_result.get('reason')}",
                                "rationale": "Evidence retrieval failed semantic validation due to topic mismatch.",
                                "sources": [],
                                "search_meta": self.search_mgr.get_search_meta(),
                                "cost": self.search_mgr.calculate_cost(gpt_model, search_type),
                                "text": fact
                            }
            
            # Fallback: Google CSE
            # If Unified Tavily search failed or returned 0 results (after filtering), try CSE.
            if not has_results and self.search_mgr.tavily_calls > 0:
                if progress_callback:
                    await progress_callback("searching_deep")
                
                # Try CSE with same query
                if primary_query:
                    cse_ctx, cse_srcs = await self.search_mgr.search_google_cse(primary_query, lang=lang)
                    if cse_ctx:
                        final_context += "\n\n=== CSE SEARCH ===\n" + cse_ctx
                    for res in cse_srcs:
                        final_sources.append({
                            "url": res.get("link"),
                            "domain": get_registrable_domain(res.get("link")),
                            "title": res.get("title"),
                            "content": res.get("snippet", ""),
                            "source_type": "general",
                            "is_trusted": False
                        })

        # 6. Analysis and Scoring
        if progress_callback:
            await progress_callback("ai_analysis")
        
        current_cost = self.search_mgr.calculate_cost(gpt_model, search_type)
        
        # Cluster (T168)
        clustered_results = None
        if claims and final_sources:
             if progress_callback:
                 await progress_callback("clustering_evidence")
             clustered_results = await self.agent.cluster_evidence(claims, final_sources)

        # T1.2: Extract anchor claim (the main claim being verified) - moved up for Social Verification
        anchor_claim = None
        if claims:
            # Find the highest importance "core" claim, fallback to first claim
            core_claims = [c for c in claims if c.get("type") == "core"]
            if core_claims:
                anchor_claim = max(core_claims, key=lambda c: c.get("importance", 0))
            else:
                anchor_claim = claims[0]

        # M67: Inline Social Verification (Tier A')
        # Check social sources for official status + content support
        if claims and final_sources:
            # Parallel verification for social sources
            verify_tasks = []
            social_indices = []
            
            for i, src in enumerate(final_sources):
                stype = src.get("source_type", "general")
                domain = src.get("domain", "")
                # Use centralized social platform registry
                if stype == "social" or is_social_platform(domain):
                    # Find relevant claim (anchor)
                    anchor = anchor_claim if anchor_claim else claims[0]
                    verify_tasks.append(self.agent.verify_social_statement(
                        anchor, 
                        src.get("content", "") or src.get("snippet", ""),
                        src.get("url", "")
                    ))
                    social_indices.append(i)
            
            if verify_tasks:
                if progress_callback:
                    await progress_callback("verifying_social")
                
                # Run verifications
                social_results = await asyncio.gather(*verify_tasks, return_exceptions=True)
                
                for idx, res in zip(social_indices, social_results):
                    if isinstance(res, dict) and res.get("tier") == "A'":
                        final_sources[idx]["evidence_tier"] = "A'"
                        final_sources[idx]["source_type"] = "official" # Critical for evidence.py ceiling
                        logger.info("[M67] Promoted Social Source %s to Tier A'", final_sources[idx].get("domain"))

        # Build Pack
        pack = build_evidence_pack(
            fact=original_fact, # Use original fact/url as the anchor
            claims=claims,
            sources=final_sources,
            search_results_clustered=clustered_results,
            content_lang=content_lang or lang,
            article_context={"text_excerpt": fact[:500]} if fact != original_fact else None
        )
        
        # M61: Signal finalizing before long LLM call to prevent UI freeze appearance
        if progress_callback:
            await progress_callback("score_evidence")
        
        # Score (T164)
        result = await self.agent.score_evidence(pack, model=gpt_model, lang=lang)
        
        # Signal finalizing after LLM call
        if progress_callback:
            await progress_callback("finalizing")
        
        # Finalize
        result["cost"] = current_cost
        # Use extracted article text for display
        # With format='text' from Tavily, `fact` contains plain text (no markdown)
        result["text"] = fact
        result["search_meta"] = self.search_mgr.get_search_meta()
        result["sources"] = enrich_sources_with_trust(final_sources)
        
        # T1.2: Assign anchor_claim to result (initialization moved up)
        if anchor_claim:
            result["anchor_claim"] = {
                "text": anchor_claim.get("text", ""),
                "type": anchor_claim.get("type", "core"),
                "importance": anchor_claim.get("importance", 1.0),
            }
        
        # Cap enforcement
        global_cap = pack.get("constraints", {}).get("global_cap", 1.0)
        verified = result.get("verified_score", -1.0)  # Sentinel: -1 means missing
        if verified < 0:
            logger.warning("[Pipeline] ⚠️ Missing verified_score in result - using 0.5")
            verified = 0.5
            result["verified_score"] = verified
        if verified > global_cap:
            result["verified_score"] = global_cap
            result["cap_applied"] = True
        
        return result

    def _extract_url_anchors(self, text: str, exclude_url: str | None = None) -> list[dict]:
        """Extract URL-anchor pairs from article text.
        
        Finds URLs with their surrounding context (anchor text) that can be
        used to identify if the URL reference survives LLM cleaning.
        
        Args:
            text: Raw article text with URLs
            exclude_url: URL to exclude (e.g., the article's own URL)
            
        Returns:
            List of dicts with 'url', 'anchor', and 'domain' keys
        """
        if not text:
            return []
        
        anchors = []
        exclude_domain = get_registrable_domain(exclude_url) if exclude_url else None
        seen_domains = set()
        
        # Pattern 1: Markdown links [anchor](url)
        md_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
        for match in re.finditer(md_pattern, text):
            anchor, url = match.groups()
            url = url.rstrip('.,;:!?')
            domain = get_registrable_domain(url)
            
            if exclude_domain and domain == exclude_domain:
                continue
            if domain in seen_domains:
                continue
            
            seen_domains.add(domain)
            anchors.append({"url": url, "anchor": anchor.strip(), "domain": domain})
        
        # Pattern 2: Parenthesized URLs like "anchor text (https://url)"
        # Common in plain text articles from extensions
        paren_pattern = r'([^(\n]{3,50})\s*\((https?://[^\s\)]+)\)'
        for match in re.finditer(paren_pattern, text):
            anchor, url = match.groups()
            url = url.rstrip('.,;:!?')
            domain = get_registrable_domain(url)
            
            if exclude_domain and domain == exclude_domain:
                continue
            if domain in seen_domains:
                continue
            
            # Clean anchor - remove leading punctuation and trailing whitespace
            anchor = anchor.strip()
            anchor = re.sub(r'^[^\w]*', '', anchor)
            if len(anchor) < 3:
                continue
            
            seen_domains.add(domain)
            anchors.append({"url": url, "anchor": anchor[:50], "domain": domain})
        
        # Pattern 3: Plain URLs - extract ~50 chars before as anchor context
        url_pattern = r'https?://[^\s\]\)\}>"\'<,]+'
        for match in re.finditer(url_pattern, text):
            url = match.group().rstrip('.,;:!?')
            domain = get_registrable_domain(url)
            
            if exclude_domain and domain == exclude_domain:
                continue
            if domain in seen_domains:
                continue
            
            # Get preceding text as anchor (up to 50 chars, stop at newline)
            start = max(0, match.start() - 60)
            prefix = text[start:match.start()]
            # Take last line/sentence fragment
            anchor = prefix.split('\n')[-1].strip()
            # Clean up
            anchor = re.sub(r'^[^\w]*', '', anchor)  # Remove leading punctuation
            if len(anchor) < 5:
                anchor = domain  # Fallback to domain name
            
            seen_domains.add(domain)
            anchors.append({"url": url, "anchor": anchor[:50], "domain": domain})
        
        return anchors[:10]  # Limit to 10 URL-anchor pairs
    
    def _restore_urls_from_anchors(self, cleaned_text: str, url_anchors: list[dict]) -> list[dict]:
        """Find which URL anchors survived cleaning and return them as sources.
        
        Args:
            cleaned_text: LLM-cleaned article text
            url_anchors: List of URL-anchor pairs from _extract_url_anchors
            
        Returns:
            List of source dicts for anchors that survived in cleaned text
        """
        if not cleaned_text or not url_anchors:
            return []
        
        sources = []
        cleaned_lower = cleaned_text.lower()
        
        for item in url_anchors:
            anchor = item.get("anchor", "")
            url = item.get("url", "")
            domain = item.get("domain", "")
            
            if not anchor or not url:
                continue
            
            # Check if anchor text (or significant part) exists in cleaned text
            anchor_lower = anchor.lower()
            # Try exact match first
            if anchor_lower in cleaned_lower:
                # Use domain as title if anchor is too short
                display_title = anchor if len(anchor) >= 10 else f"Джерело: {domain}"
                sources.append({
                    "url": url,
                    "title": display_title,
                    "domain": domain,
                    "source_type": "inline",
                    "is_trusted": False
                })
                continue
            
            # Try fuzzy: check if most words from anchor appear
            anchor_words = [w for w in anchor_lower.split() if len(w) > 3]
            if anchor_words:
                matches = sum(1 for w in anchor_words if w in cleaned_lower)
                if matches >= len(anchor_words) * 0.7:  # 70% word match
                    display_title = anchor if len(anchor) >= 10 else f"Джерело: {domain}"
                    sources.append({
                        "url": url,
                        "title": display_title,
                        "domain": domain,
                        "source_type": "inline",
                        "is_trusted": False
                    })
        
        return sources[:5]  # Limit to 5 inline sources


    def _is_url(self, text: str) -> bool:
        # Text must START with http:// or https:// to be treated as URL input
        # This prevents text with inline URLs from being treated as URL input
        if not text or len(text) > 500:
            return False
        stripped = text.strip()
        return stripped.startswith("http://") or stripped.startswith("https://")

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

    # ─────────────────────────────────────────────────────────────────────────
    # M64: Topic-Aware Round-Robin Query Selection ("Coverage Engine")
    # ─────────────────────────────────────────────────────────────────────────
    
    def _select_diverse_queries(
        self, 
        claims: list, 
        max_queries: int = 3,
        fact_fallback: str = ""
    ) -> list[str]:
        """
        M64: Topic-Aware Round-Robin Query Selection ("Coverage Engine").
        
        Ensures every topic_key gets at least 1 query before any topic gets
        a 2nd query. This solves the "Digest Problem" where multi-topic 
        articles had all queries spent on the first topic only.
        
        Algorithm:
        1. PREPROCESS: Filter sidefacts, group by topic_key, sort by importance
        2. PASS 1 (COVERAGE): 1 CORE query per topic_key (round-robin)
        3. PASS 2 (DEPTH): NUMERIC/ATTRIBUTION if budget remains
        4. PASS 3 (FILL): LOCAL/remaining queries
        5. POST: Fuzzy dedup (90%)
        
        Args:
            claims: List of Claim objects with topic_key, query_candidates
            max_queries: Maximum queries to return
            fact_fallback: Fallback text if no claims/queries available
            
        Returns:
            List of diverse search queries (max max_queries)
        """
        from collections import defaultdict
        
        if not claims:
            return [normalize_search_query(fact_fallback[:200])] if fact_fallback else []
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. PREPROCESS: Filter and group claims
        # ─────────────────────────────────────────────────────────────────────
        MIN_WORTHINESS = 0.4
        eligible = [
            c for c in claims 
            if c.get("type") != "sidefact"  # Skip sidefacts entirely
            and c.get("check_worthiness", c.get("importance", 0.5)) >= MIN_WORTHINESS
        ]
        
        if not eligible:
            # Fallback: take highest importance claim even if below threshold
            eligible = sorted(
                [c for c in claims if c.get("type") != "sidefact"],
                key=lambda c: c.get("importance", 0), 
                reverse=True
            )[:1]
            if not eligible:
                eligible = claims[:1]
            logger.info("[M64] All claims below threshold or sidefacts, using highest importance")
        
        # Group by topic_key (specific entity) for round-robin coverage
        # If topic_key missing, fallback to topic_group
        groups: dict[str, list] = defaultdict(list)
        for c in eligible:
            key = c.get("topic_key") or c.get("topic_group", "Other")
            groups[key].append(c)
        
        # M73: Calculate effective score for sorting
        # Combines: importance, structural weight, tension score, and key_claim flag
        def _effective_score(claim: dict) -> float:
            base_importance = claim.get("importance", 0.5)
            # M73 Layer 2: Bonus for structurally important claims
            structural_bonus = min(0.1, claim.get("graph_structural_weight", 0) * 0.1)
            # M73 Layer 3: Bonus for high-tension claims (need verification)
            tension_bonus = min(0.15, claim.get("graph_tension_score", 0) * 0.15)
            # M73: Key claims get strong priority
            key_claim_bonus = 0.2 if claim.get("is_key_claim") else 0.0
            return base_importance + structural_bonus + tension_bonus + key_claim_bonus
        
        # Sort groups by max effective score of their claims (most important topics first)
        sorted_keys = sorted(
            groups.keys(),
            key=lambda k: max(_effective_score(c) for c in groups[k]),
            reverse=True
        )
        
        # Sort claims within each group by effective score
        for key in groups:
            groups[key].sort(key=_effective_score, reverse=True)
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. PASS 1: COVERAGE (Critical) - One CORE query per topic
        # ─────────────────────────────────────────────────────────────────────
        selected: list[str] = []
        covered_topics: set[str] = set()  # Track which topics have at least 1 query
        
        for key in sorted_keys:
            if len(selected) >= max_queries:
                break
            
            top_claim = groups[key][0]
            core_query = self._get_query_by_role(top_claim, "CORE")
            
            if core_query:
                normalized = self._normalize_and_sanitize(core_query)
                if normalized and not self._is_fuzzy_duplicate(normalized, selected, threshold=0.9):
                    selected.append(normalized)
                    covered_topics.add(key)
                    logger.debug("[M64] Pass 1 (Coverage): topic=%s, query=%s", key, normalized[:50])
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. PASS 2: DEPTH - NUMERIC/ATTRIBUTION for already-covered topics
        # ─────────────────────────────────────────────────────────────────────
        if len(selected) < max_queries:
            for key in sorted_keys:
                if len(selected) >= max_queries:
                    break
                if key not in covered_topics:
                    continue  # Only add depth to already-covered topics
                
                top_claim = groups[key][0]
                for role in ["NUMERIC", "ATTRIBUTION"]:
                    query = self._get_query_by_role(top_claim, role)
                    if query:
                        normalized = self._normalize_and_sanitize(query)
                        if normalized and not self._is_fuzzy_duplicate(normalized, selected, threshold=0.9):
                            selected.append(normalized)
                            logger.debug("[M64] Pass 2 (Depth): topic=%s, role=%s, query=%s", 
                                        key, role, normalized[:50])
                            break  # One depth query per topic
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. PASS 3: FILL - Any remaining valid queries
        # ─────────────────────────────────────────────────────────────────────
        if len(selected) < max_queries:
            for key in sorted_keys:
                if len(selected) >= max_queries:
                    break
                for claim in groups[key]:
                    if len(selected) >= max_queries:
                        break
                    # Try all queries from query_candidates
                    for candidate in claim.get("query_candidates", []):
                        query = candidate.get("text", "")
                        normalized = self._normalize_and_sanitize(query)
                        if normalized and not self._is_fuzzy_duplicate(normalized, selected, threshold=0.9):
                            selected.append(normalized)
                            logger.debug("[M64] Pass 3 (Fill): topic=%s, query=%s", key, normalized[:50])
                            if len(selected) >= max_queries:
                                break
                    # Also try legacy search_queries
                    for query in claim.get("search_queries", []):
                        normalized = self._normalize_and_sanitize(query)
                        if normalized and not self._is_fuzzy_duplicate(normalized, selected, threshold=0.9):
                            selected.append(normalized)
                            logger.debug("[M64] Pass 3 (Fill/Legacy): topic=%s, query=%s", key, normalized[:50])
                            if len(selected) >= max_queries:
                                break
        
        # ─────────────────────────────────────────────────────────────────────
        # 5. FALLBACK: If no queries selected, use fact_fallback
        # ─────────────────────────────────────────────────────────────────────
        if not selected and fact_fallback:
            selected = [normalize_search_query(fact_fallback[:200])]
        
        # Log summary
        logger.info(
            "[M64] Query selection: %d eligible claims, %d topic_keys -> %d queries. Topics covered: %s",
            len(eligible), len(groups), len(selected), list(covered_topics)[:5]
        )
        
        return selected[:max_queries]
    
    def _get_query_by_role(self, claim: dict, role: str) -> str | None:
        """
        M64: Extract query with specific role from claim's query_candidates.
        
        Args:
            claim: Claim dict with query_candidates and/or search_queries
            role: Query role ("CORE", "NUMERIC", "ATTRIBUTION", "LOCAL")
            
        Returns:
            Query text or None if not found
        """
        # Try query_candidates first (M64 format)
        for candidate in claim.get("query_candidates", []):
            if candidate.get("role") == role:
                return candidate.get("text", "")
        
        # Fallback to legacy search_queries by position
        legacy = claim.get("search_queries", [])
        role_index = {"CORE": 0, "NUMERIC": 1, "ATTRIBUTION": 2, "LOCAL": 0}
        idx = role_index.get(role, 0)
        if idx < len(legacy):
            return legacy[idx]
        # If role not found but legacy exists, return first as CORE fallback
        if role == "CORE" and legacy:
            return legacy[0]
        return None
    
    def _normalize_and_sanitize(self, query: str) -> str | None:
        """
        M64: Normalize query.
        
        Note: Strict gambling keywords removal is deprecated (M64).
        We rely on LLM constraints and Tavily 'topic="news"' mode 
        to prevent gambling/spam results instead of hardcoded stoplists.
        
        Args:
            query: Raw query text
            
        Returns:
            Normalized query or None if invalid
        """
        if not query:
            return None
        
        normalized = normalize_search_query(query)
        if not normalized:
            return None
        
        return normalized
    
    def _is_fuzzy_duplicate(self, query: str, existing: list[str], threshold: float = 0.9) -> bool:
        """
        M64: Check if query is >threshold similar to any existing query.
        
        Uses Jaccard similarity on word sets.
        
        Args:
            query: Query to check
            existing: List of already-selected queries
            threshold: Similarity threshold (0.9 = 90% word overlap)
            
        Returns:
            True if query is a duplicate
        """
        query_words = set(query.lower().split())
        if not query_words:
            return True  # Empty query is always a "duplicate"
        
        for existing_query in existing:
            existing_words = set(existing_query.lower().split())
            if not existing_words:
                continue
            
            # Jaccard similarity: |intersection| / |union|
            intersection = len(query_words & existing_words)
            union = len(query_words | existing_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                logger.debug("[M64] Fuzzy duplicate (%.0f%%): '%s' ≈ '%s'", 
                            similarity * 100, query[:30], existing_query[:30])
                return True
        
        return False


    def _can_add_search(self, model, search_type, max_cost):
        # Speculative cost check
        current = self.search_mgr.calculate_cost(model, search_type)
        # Add cost of 1 search
        step_cost = int(SEARCH_COSTS.get(search_type, 80))
        return self.search_mgr.can_afford(current + step_cost, max_cost)

    def _finalize_oracle(self, oracle_res: dict, fact: str) -> dict:
        """Format oracle result for return (legacy)."""
        oracle_res["text"] = fact
        oracle_res["search_meta"] = self.search_mgr.get_search_meta()
        return oracle_res

    async def _finalize_oracle_hybrid(
        self, 
        oracle_result: dict, 
        fact: str, 
        lang: str = "en", 
        progress_callback=None
    ) -> dict:
        """
        M63: Format Oracle JACKPOT result for immediate return.
        M67: Added lang parameter for localization support.
        M69: Added granular progress updates (localizing_content).
        
        Converts OracleCheckResult to FactCheckResponse format.
        """
        oracle_result.get("status", "MIXED")
        rating = oracle_result.get("rating", "")
        publisher = oracle_result.get("publisher", "Fact Check")
        url = oracle_result.get("url", "")
        claim_reviewed = oracle_result.get("claim_reviewed", "")
        summary = oracle_result.get("summary", "")
        
        # M67: Use LLM-determined scores from OracleValidationSkill (no heuristics!)
        verified_score = float(oracle_result.get("verified_score", -1.0))
        danger_score = float(oracle_result.get("danger_score", -1.0))
        
        # Warning only - no fallback! If -1, it's a bug and should be visible.
        if verified_score < 0 or danger_score < 0:
            logger.warning("[Pipeline] ⚠️ Oracle result missing LLM scores. BUG!")
        
        # Build response (English first)
        analysis = f"According to {publisher}, this claim is rated as '{rating}'. {summary}"
        rationale = f"Fact check by {publisher}: Rated as '{rating}'. {claim_reviewed}"
        
        # M67: Translate if non-English and translation_service available
        if lang and lang.lower() not in ("en", "en-us") and self.translation_service:
            if progress_callback:
                await progress_callback("localizing_content")
            
            try:
                analysis = await self.translation_service.translate(analysis, target_lang=lang)
                rationale = await self.translation_service.translate(rationale, target_lang=lang)
            except Exception as e:
                logger.warning("[Pipeline] Translation failed for Oracle result: %s", e)
                # Keep English if translation fails
        
        sources = [{
            "title": f"Fact Check by {publisher}",
            "link": url,
            "url": url,
            "domain": get_registrable_domain(url) if url else publisher.lower().replace(" ", ""),
            "snippet": f"Rating: {rating}. {summary}",
            "origin": "GOOGLE_FACT_CHECK",
            "source_type": "fact_check",
            "is_trusted": True,
        }]
        
        return {
            "verified_score": verified_score,
            "confidence_score": 1.0,
            "danger_score": danger_score,
            "context_score": 1.0,
            "style_score": 1.0,
            "analysis": analysis,
            "rationale": rationale,
            "sources": sources,
            "cost": 0,  # Oracle is free!
            "rgba": [danger_score, verified_score, 1.0, 1.0],
            "text": fact,
            "search_meta": self.search_mgr.get_search_meta(),
            "oracle_jackpot": True,  # M63: Flag for frontend
        }

    def _create_oracle_source(self, oracle_result: dict) -> dict:
        """
        M63: Create source dict from Oracle result for EVIDENCE scenario.
        
        This source is added to the evidence pack as a Tier A (high trust) source.
        """
        url = oracle_result.get("url", "")
        publisher = oracle_result.get("publisher", "Fact Check")
        rating = oracle_result.get("rating", "")
        claim_reviewed = oracle_result.get("claim_reviewed", "")
        summary = oracle_result.get("summary", "")
        relevance = oracle_result.get("relevance_score", 0.0)
        status = oracle_result.get("status", "MIXED")
        
        return {
            "url": url,
            "domain": get_registrable_domain(url) if url else publisher.lower().replace(" ", ""),
            "title": f"Fact Check: {claim_reviewed[:50]}..." if len(claim_reviewed) > 50 else f"Fact Check: {claim_reviewed}",
            "content": f"{publisher} rated this claim as '{rating}': {summary}",
            "snippet": f"Rating: {rating}. {summary[:200]}",
            "source_type": "fact_check",
            "is_trusted": True,
            "origin": "GOOGLE_FACT_CHECK",
            # M63: Oracle metadata for transparency in scoring
            "oracle_metadata": {
                "relevance_score": relevance,
                "status": status,
                "publisher": publisher,
                "rating": rating,
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    # M70: Schema-First Query Generation (Assertion-Based)
    # ─────────────────────────────────────────────────────────────────────────

    def _select_queries_from_claim_units(
        self,
        claim_units: list,
        max_queries: int = 3,
        fact_fallback: str = "",
    ) -> list[str]:
        """
        M70: Generate search queries from structured ClaimUnits.
        
        Key difference from legacy:
        - Only FACT assertions generate verification queries
        - CONTEXT assertions are informational (no refutation search)
        - Queries are built from assertion_key + value
        
        Args:
            claim_units: List of ClaimUnit objects (from schema)
            max_queries: Maximum queries to return
            fact_fallback: Fallback text if no queries generated
            
        Returns:
            List of search queries
        """
        from spectrue_core.schema import ClaimUnit
        
        if not claim_units:
            return [normalize_search_query(fact_fallback[:200])] if fact_fallback else []
        
        queries: list[str] = []
        
        for unit in claim_units:
            if not isinstance(unit, ClaimUnit):
                continue
                
            # Only FACT assertions get verification queries
            fact_assertions = unit.get_fact_assertions()
            
            for assertion in fact_assertions:
                if len(queries) >= max_queries:
                    break
                    
                query = self._build_assertion_query(unit, assertion)
                if query:
                    normalized = self._normalize_and_sanitize(query)
                    if normalized and not self._is_fuzzy_duplicate(normalized, queries, threshold=0.9):
                        queries.append(normalized)
                        logger.debug(
                            "[M70] Assertion query: key=%s, query=%s",
                            assertion.key, normalized[:50]
                        )
        
        # Fallback if no queries generated
        if not queries:
            # Try to use claim text
            for unit in claim_units:
                if isinstance(unit, ClaimUnit) and unit.text:
                    return [normalize_search_query(unit.text[:200])]
            # Last resort: fact_fallback
            if fact_fallback:
                return [normalize_search_query(fact_fallback[:200])]
        
        logger.info(
            "[M70] Query selection: %d claims, %d FACT queries generated",
            len(claim_units), len(queries)
        )
        
        return queries[:max_queries]

    def _build_assertion_query(self, unit, assertion) -> str | None:
        """
        M70: Build search query for a specific assertion.
        
        Query structure: "{subject} {assertion.value} {context}"
        
        Examples:
        - event.location.city: "Joshua Paul fight Miami official location"
        - numeric.value: "Bitcoin price $42000 official"
        - event.time: "Joshua Paul fight March 2025 date confirmed"
        
        Args:
            unit: ClaimUnit containing the assertion
            assertion: Assertion to build query for
            
        Returns:
            Search query string or None
        """
        from spectrue_core.schema import Assertion, Dimension
        
        if not isinstance(assertion, Assertion):
            return None
            
        # Don't generate queries for CONTEXT (they're informational)
        if assertion.dimension == Dimension.CONTEXT:
            return None
        
        parts: list[str] = []
        
        # Add subject if available
        if unit.subject:
            parts.append(unit.subject)
        
        # Add object if available (often the other party)
        if unit.object:
            parts.append(unit.object)
        
        # Add assertion value
        if assertion.value:
            value_str = str(assertion.value)
            if len(value_str) < 50:  # Don't add very long values
                parts.append(value_str)
        
        # Add context based on assertion key
        key = assertion.key
        if "location" in key:
            parts.append("location official")
        elif "time" in key or "date" in key:
            parts.append("date confirmed")
        elif "quote" in key or "attribution" in key:
            parts.append("said statement")
        elif "numeric" in key or "value" in key:
            parts.append("official data")
        else:
            parts.append("verified")
        
        if not parts:
            return None
            
        return " ".join(parts)

    def _get_claim_units_for_evidence_mapping(
        self,
        claim_units: list,
        sources: list[dict],
    ) -> dict[str, list[str]]:
        """
        M70: Map sources to assertion_keys for targeted verification.
        
        This is used by clustering to understand which assertion
        each piece of evidence relates to.
        
        Returns:
            Dict of claim_id -> list of assertion_keys that need evidence
        """
        from spectrue_core.schema import ClaimUnit, Dimension
        
        mapping: dict[str, list[str]] = {}
        
        for unit in claim_units:
            if not isinstance(unit, ClaimUnit):
                continue
                
            fact_keys = [
                a.key for a in unit.assertions
                if a.dimension == Dimension.FACT
            ]
            
            if fact_keys:
                mapping[unit.id] = fact_keys
        
        return mapping
