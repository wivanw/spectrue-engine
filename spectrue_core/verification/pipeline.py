from spectrue_core.verification.search_mgr import SearchManager, SEARCH_COSTS
from spectrue_core.verification.evidence import build_evidence_pack
from spectrue_core.verification.trusted_sources import get_trusted_domains_by_lang
from spectrue_core.utils.text_processing import clean_article_text, normalize_search_query
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.utils.trust_utils import enrich_sources_with_trust
from spectrue_core.utils.trace import Trace
from spectrue_core.config import SpectrueConfig
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.agents.skills.oracle_validation import EVIDENCE_THRESHOLD
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
    def __init__(self, config: SpectrueConfig, agent: FactCheckerAgent):
        self.config = config
        self.agent = agent
        # M63: Pass oracle_skill to SearchManager for hybrid mode
        self.search_mgr = SearchManager(config, oracle_validator=agent.oracle_skill)

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
                # Prefer normalized_text over raw text for cleaner Oracle query
                query_text = cand.get("normalized_text") or cand.get("text", "")
                q = normalize_search_query(query_text)
                
                logger.info("[Pipeline] Oracle Hybrid Check: intent=%s, query=%s", article_intent, q[:50])
                
                # M63: Use new hybrid method with LLM validation
                oracle_result = await self.search_mgr.check_oracle_hybrid(q, intent=article_intent)
                
                relevance = oracle_result.get("relevance_score", 0.0)
                status = oracle_result.get("status", "EMPTY")
                is_jackpot = oracle_result.get("is_jackpot", False)
                
                Trace.event("pipeline.oracle_hybrid", {
                    "intent": article_intent,
                    "query": q[:50],
                    "relevance_score": relevance,
                    "status": status,
                    "is_jackpot": is_jackpot
                })
                
                if status == "EMPTY":
                    logger.info("[Pipeline] Oracle: No results found. Continuing to search.")
                    continue
                
                # SCENARIO A: JACKPOT (relevance > 0.9)
                if is_jackpot:
                    logger.info(
                        "[Pipeline] 🎰 JACKPOT! Oracle hit (score=%.2f). Stopping pipeline.",
                        relevance
                    )
                    return self._finalize_oracle_hybrid(oracle_result, original_fact)
                
                # SCENARIO B: EVIDENCE (0.5 < relevance <= 0.9)
                elif relevance > EVIDENCE_THRESHOLD:
                    logger.info(
                        "[Pipeline] 📚 EVIDENCE: Oracle related (score=%.2f). Adding to pack.",
                        relevance
                    )
                    # Create source from Oracle result for evidence pack
                    oracle_evidence_source = self._create_oracle_source(oracle_result)
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

        # M62: Smart Query Selection (typed priority slots)
        search_queries = self._select_diverse_queries(claims, max_queries=3, fact_fallback=fact)
        
        # 5. Parallel Waterfall Search (Tier 1 + Tier 2)
        if not preloaded_context:
            if progress_callback:
                await progress_callback("searching_tier1")
            
            search_lang = content_lang or lang
            tier1_domains = get_trusted_domains_by_lang(search_lang)
            
            # Select queries
            # Select queries
            # T1 (Trusted) prefers Event-Based (Index 1) for broader context in trusted sources
            # Fallback to Specific (Index 0) if only 1 query exists
            if len(search_queries) > 1:
                t1_query = search_queries[1]
            else:
                t1_query = search_queries[0]
            
            # T2 (General) uses Local query (Index 2) if available to capture local context
            # Fallback to Specific (Index 0)
            if len(search_queries) > 2:
                t2_query = search_queries[2]
            else:
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
            
            
            # T9: Simplified "Smart" Mode - Waterfall Strategy
            # Always run T1 first. If T1 results are insufficient (< 2), fall back to T2.
            # Parallel execution removed to save quota and reduce noise.
            
            tasks = []
            
            # Task T1
            if self._can_add_search(gpt_model, search_type, max_cost):
                tasks.append(self.search_mgr.search_tier1(t1_query, tier1_domains))
            
            # Execute T1
            results = await asyncio.gather(*tasks)
            
            # Process T1
            if len(results) > 0:
                ctx1, srcs1 = results[0]
                final_context += "\n" + ctx1
                final_sources.extend(srcs1)
            
            run_t2_parallel = False  # Always waterfall in smart mode
            
            # Waterfall Fallback (if T2 didn't run parallel AND T1 was weak)
            # M60: Count only search sources, not inline sources from article
            search_sources_count = len([s for s in final_sources if s.get("source_type") != "inline"])
            
            # T8: Ensure minimum 3 unique domains for diversity (increased from 2)
            if not run_t2_parallel and search_sources_count < 3 and self._can_add_search(gpt_model, search_type, max_cost):
                if progress_callback:
                    await progress_callback("searching_tier2_fallback")
                # Now we know T1 domains, but we can just exclude the predefined TRUSTED list to be safe/consistent
                ctx2, srcs2 = await self.search_mgr.search_tier2(t2_query, exclude_domains=tier1_domains)
                if srcs2:
                    final_context += "\n\n=== GENERAL SEARCH ===\n" + ctx2
                    final_sources.extend(srcs2)
                else:
                    # CSE Fallback (Tier 3)
                     if self.search_mgr.tavily_calls > 0:
                         if progress_callback:
                             await progress_callback("searching_deep")
                         cse_ctx, cse_srcs = await self.search_mgr.search_google_cse(t1_query, lang=lang)
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
             clustered_results = await self.agent.cluster_evidence(claims, final_sources)

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
            await progress_callback("finalizing")
        
        # Score (T164)
        result = await self.agent.score_evidence(pack, model=gpt_model, lang=lang)
        
        # Finalize
        result["cost"] = current_cost
        # Use extracted article text for display
        # With format='text' from Tavily, `fact` contains plain text (no markdown)
        result["text"] = fact
        result["search_meta"] = self.search_mgr.get_search_meta()
        result["sources"] = enrich_sources_with_trust(final_sources)
        
        # T1.2: Extract anchor claim (the main claim being verified)
        anchor_claim = None
        if claims:
            # Find the highest importance "core" claim, fallback to first claim
            core_claims = [c for c in claims if c.get("type") == "core"]
            if core_claims:
                anchor_claim = max(core_claims, key=lambda c: c.get("importance", 0))
            else:
                anchor_claim = claims[0]
        if anchor_claim:
            result["anchor_claim"] = {
                "text": anchor_claim.get("text", ""),
                "type": anchor_claim.get("type", "core"),
                "importance": anchor_claim.get("importance", 1.0),
            }
        
        # Cap enforcement
        global_cap = pack.get("constraints", {}).get("global_cap", 1.0)
        verified = result.get("verified_score", 0.5)
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

    def _select_diverse_queries(
        self, 
        claims: list, 
        max_queries: int = 3,
        fact_fallback: str = ""
    ) -> list[str]:
        """
        M62: Typed Priority Slot Query Selection.
        
        Instead of round-robin across topics, we use typed priority slots:
        - Slot 1: Core Claim (Event query) - most important
        - Slot 2: Numeric/Timeline Claim (if available)
        - Slot 3: Attribution/Quote Claim (if available)
        - Slot 4+: Next Core from different topic_group (if budget allows)
        
        Sidefacts are SKIPPED entirely to save search budget.
        
        Args:
            claims: List of Claim objects with type, topic_group, check_worthiness
            max_queries: Maximum queries to return
            fact_fallback: Fallback text if no claims/queries available
            
        Returns:
            List of diverse search queries (max max_queries)
        """
        if not claims:
            return [normalize_search_query(fact_fallback[:200])] if fact_fallback else []
        
        # 1. Filter claims
        MIN_WORTHINESS = 0.4
        eligible_claims = [
            c for c in claims 
            if c.get("type") != "sidefact"  # Skip sidefacts entirely
            and c.get("check_worthiness", c.get("importance", 0.5)) >= MIN_WORTHINESS
        ]
        
        if not eligible_claims:
            # Fallback: take highest importance claim even if below threshold
            eligible_claims = sorted(
                [c for c in claims if c.get("type") != "sidefact"],
                key=lambda c: c.get("importance", 0), 
                reverse=True
            )[:1]
            if not eligible_claims:
                eligible_claims = claims[:1]
            logger.info("[M62] All claims below threshold or sidefacts, using highest importance")
        
        # 2. Categorize by type
        core_claims = [c for c in eligible_claims if c.get("type") == "core"]
        numeric_timeline = [c for c in eligible_claims if c.get("type") in ("numeric", "timeline")]
        attribution = [c for c in eligible_claims if c.get("type") == "attribution"]
        
        # Sort each category by importance (descending)
        core_claims.sort(key=lambda c: c.get("importance", 0), reverse=True)
        numeric_timeline.sort(key=lambda c: c.get("importance", 0), reverse=True)
        attribution.sort(key=lambda c: c.get("importance", 0), reverse=True)
        
        # 3. Build priority slot queue
        slot_queue: list[dict] = []
        
        # Slot 1: Core Claim (mandatory if exists)
        if core_claims:
            slot_queue.append(core_claims[0])
        
        # Slot 2: Numeric/Timeline Claim (if exists)
        if numeric_timeline:
            slot_queue.append(numeric_timeline[0])
        
        # Slot 3: Attribution Claim (if exists)
        if attribution:
            slot_queue.append(attribution[0])
        
        # Slot 4+: Additional Core claims from DIFFERENT topic groups
        if len(slot_queue) < max_queries and len(core_claims) > 1:
            seen_topics = {slot_queue[0].get("topic_group", "Other")} if slot_queue else set()
            
            for c in core_claims[1:]:
                if len(slot_queue) >= max_queries:
                    break
                topic = c.get("topic_group", "Other")
                if topic not in seen_topics:
                    slot_queue.append(c)
                    seen_topics.add(topic)
        
        # 4. Extract queries from slot queue
        selected_queries: list[str] = []
        seen_queries: set[str] = set()
        
        for claim in slot_queue[:max_queries]:
            queries = claim.get("search_queries", [])
            
            # Use Event-Based query (index 1) as primary, fallback to index 0
            query_idx = 1 if len(queries) > 1 else 0
            if query_idx < len(queries):
                raw_query = queries[query_idx]
                normalized = normalize_search_query(raw_query)
                
                if normalized and normalized not in seen_queries:
                    selected_queries.append(normalized)
                    seen_queries.add(normalized)
                    claim_type = claim.get("type", "?")
                    claim_topic = claim.get("topic_group", "?")
                    logger.debug(
                        "[M62] Slot %d: type=%s, topic=%s, query=%s",
                        len(selected_queries), claim_type, claim_topic, normalized[:50]
                    )
        
        # 5. Fill remaining slots if we have fewer queries than slots
        if len(selected_queries) < max_queries:
            # Try to get any remaining queries from eligible claims
            for c in eligible_claims:
                if len(selected_queries) >= max_queries:
                    break
                for q in c.get("search_queries", []):
                    if len(selected_queries) >= max_queries:
                        break
                    normalized = normalize_search_query(q)
                    if normalized and normalized not in seen_queries:
                        selected_queries.append(normalized)
                        seen_queries.add(normalized)
        
        # 6. Fallback if no queries selected
        if not selected_queries:
            if fact_fallback:
                selected_queries = [normalize_search_query(fact_fallback[:200])]
        
        # Log summary
        slot_types = [c.get("type", "?") for c in slot_queue[:len(selected_queries)]]
        logger.info(
            "[M62] Query selection: %d eligible claims -> %d queries. Slot types: %s",
            len(eligible_claims), len(selected_queries), slot_types
        )
        
        return selected_queries[:max_queries]

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

    def _finalize_oracle_hybrid(self, oracle_result: dict, fact: str) -> dict:
        """
        M63: Format Oracle JACKPOT result for immediate return.
        
        Converts OracleCheckResult to FactCheckResponse format.
        """
        status = oracle_result.get("status", "MIXED")
        rating = oracle_result.get("rating", "")
        publisher = oracle_result.get("publisher", "Fact Check")
        url = oracle_result.get("url", "")
        claim_reviewed = oracle_result.get("claim_reviewed", "")
        summary = oracle_result.get("summary", "")
        
        # Map status to verified_score
        if status == "CONFIRMED":
            verified_score = 0.9
            danger_score = 0.1
        elif status == "REFUTED":
            verified_score = 0.1
            danger_score = 0.9
        else:  # MIXED
            verified_score = 0.4
            danger_score = 0.6
        
        # Build response
        analysis = f"According to {publisher}, this claim is rated as '{rating}'. {summary}"
        rationale = f"Fact check by {publisher}: Rated as '{rating}'. {claim_reviewed}"
        
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

