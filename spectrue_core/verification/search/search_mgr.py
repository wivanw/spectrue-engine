# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.tools.web_search_tool import WebSearchTool
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool
from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.evidence.evidence_pack import OracleCheckResult
from spectrue_core.verification.evidence.evidence import (
    needs_evidence_acquisition_ladder,
    extract_quote_candidates,
)
from spectrue_core.verification.types import SearchResponse
from spectrue_core.verification.search.search_policy import (
    build_context_from_sources,
    filter_search_results,
    rerank_search_results,
    prefer_fallback_results,
    should_fallback_news_to_general,
    SearchPolicyProfile,
)
from spectrue_core.scoring.budget_allocation import GlobalBudgetTracker
from spectrue_core.utils.trace import Trace
import logging

logger = logging.getLogger(__name__)

SKIP_EXTENSIONS = (".txt", ".xml", ".zip")

class SearchManager:
    """
    Manages search tools, execution, and cost budgeting.
    """
    def __init__(self, config: SpectrueConfig, oracle_validator=None):
        """
        Initialize SearchManager.
        
        Args:
            config: Spectrue configuration
            oracle_validator: Optional OracleValidationSkill for hybrid mode
        """
        self.config = config
        self.web_tool = WebSearchTool(config)
        # Pass validator to GoogleFactCheckTool for hybrid mode
        self.oracle_tool = GoogleFactCheckTool(config, oracle_validator=oracle_validator)
        self.cse_tool = GoogleCSESearchTool(config)

        # Metrics state for current run
        self.tavily_calls = 0
        self.google_cse_calls = 0
        self.page_fetches = 0
        self.oracle_calls = 0  # Track Oracle API calls
        
        # Separate budget trackers for different extraction contexts
        # Inline sources use a single tracker (processed once before claims)
        self.inline_budget_tracker = GlobalBudgetTracker()  # For inline source verification
        
        # Per-claim budget trackers (each claim gets independent budget in deep mode)
        self._claim_budget_trackers: dict[str, GlobalBudgetTracker] = {}
        
        # Backward compatibility alias (creates default tracker on demand)
        self._default_claim_tracker: GlobalBudgetTracker | None = None
        
        self.policy_profile: SearchPolicyProfile | None = None
    
    def get_claim_budget_tracker(self, claim_id: str | None = None) -> GlobalBudgetTracker:
        """Get budget tracker for a specific claim (or default if no claim_id)."""
        if claim_id is None:
            # Backward compatibility: use a single default tracker
            if self._default_claim_tracker is None:
                self._default_claim_tracker = GlobalBudgetTracker()
            return self._default_claim_tracker
        
        # Per-claim tracker: create if doesn't exist
        if claim_id not in self._claim_budget_trackers:
            self._claim_budget_trackers[claim_id] = GlobalBudgetTracker()
        return self._claim_budget_trackers[claim_id]
    
    @property
    def budget_tracker(self) -> GlobalBudgetTracker:
        """Backward compatibility: returns default claim tracker."""
        return self.get_claim_budget_tracker(None)
    
    @property
    def claim_budget_tracker(self) -> GlobalBudgetTracker:
        """Backward compatibility alias."""
        return self.get_claim_budget_tracker(None)

    def reset_metrics(self):
        self.tavily_calls = 0
        self.google_cse_calls = 0
        self.page_fetches = 0
        self.oracle_calls = 0
        self.inline_budget_tracker = GlobalBudgetTracker()  # Reset inline budget
        self._claim_budget_trackers = {}  # Reset all per-claim budgets
        self._default_claim_tracker = None  # Reset default tracker
        self.policy_profile = None

    def set_policy_profile(self, profile: SearchPolicyProfile | None) -> None:
        self.policy_profile = profile

    def decide_retrieval_action(
        self,
        *,
        retrieval_eval: dict | None = None,
        retrieval_confidence: float | None = None,
        claim: dict | None = None,
    ) -> tuple[str, str]:
        """
        Decide stop/continue/correction action for retrieval evaluation.
        """
        if retrieval_eval is not None and retrieval_confidence is None:
            try:
                retrieval_confidence = float(retrieval_eval.get("retrieval_confidence", 0.0))
            except Exception:
                retrieval_confidence = 0.0
        retrieval_confidence = float(retrieval_confidence or 0.0)

        expected_gain = None
        expected_cost = None
        value_per_cost = None
        if isinstance(retrieval_eval, dict):
            expected_gain = retrieval_eval.get("expected_gain")
            expected_cost = retrieval_eval.get("expected_cost")
            value_per_cost = retrieval_eval.get("value_per_cost")

        calibration = getattr(getattr(self.config, "runtime", None), "calibration", None)
        min_value_per_cost = float(getattr(calibration, "retrieval_min_value_per_cost", 0.25) or 0.25)
        gain_floor = float(getattr(calibration, "retrieval_gain_floor", 0.15) or 0.15)
        cost_weight = float(getattr(calibration, "retrieval_cost_weight", 1.0) or 1.0)

        if expected_gain is not None:
            try:
                expected_gain = float(expected_gain)
            except Exception:
                expected_gain = None
        if expected_cost is not None:
            try:
                expected_cost = float(expected_cost)
            except Exception:
                expected_cost = None
        if value_per_cost is None and expected_gain is not None and expected_cost is not None:
            denom = max(1e-6, expected_cost * max(1e-6, cost_weight))
            value_per_cost = expected_gain / denom

        if expected_gain is not None and value_per_cost is not None:
            if expected_gain <= gain_floor or value_per_cost < min_value_per_cost:
                return "stop_early", "marginal_gain_below_cost"

        calibration = getattr(getattr(self.config, "runtime", None), "calibration", None)
        low_threshold = float(getattr(calibration, "retrieval_confidence_low", 0.35) or 0.35)
        high_threshold = float(getattr(calibration, "retrieval_confidence_high", 0.70) or 0.70)
        guard_applied = False
        if self.policy_profile is not None:
            base = float(self.policy_profile.quality_thresholds.min_relevance_score or 0.0)
            if base > 0:
                low_threshold = max(low_threshold, base)
                high_threshold = max(high_threshold, base)
        if high_threshold <= low_threshold:
            high_threshold = min(1.0, low_threshold + 0.05)
            guard_applied = True
        if guard_applied:
            Trace.event(
                "retrieval.threshold_guard_applied",
                {
                    "low_threshold": float(low_threshold),
                    "high_threshold": float(high_threshold),
                    "policy_profile_enabled": self.policy_profile is not None,
                    "base_min_relevance": float(
                        getattr(getattr(self.policy_profile, "quality_thresholds", None), "min_relevance_score", 0.0)
                        or 0.0
                    )
                    if self.policy_profile is not None
                    else 0.0,
                },
            )

        if retrieval_confidence >= high_threshold:
            return "stop_early", "confidence_high"

        if retrieval_confidence <= low_threshold:
            metadata = claim.get("metadata") if isinstance(claim, dict) else None
            tension = 0.0
            if isinstance(claim, dict):
                try:
                    tension = float(claim.get("graph_tension_score") or 0.0)
                except Exception:
                    tension = 0.0
            if tension >= 0.6:
                return "change_channel", "low_confidence_high_tension"
            locale_plan = getattr(metadata, "search_locale_plan", None) if metadata else None
            fallbacks = list(getattr(locale_plan, "fallback", []) or [])
            if fallbacks:
                return "change_language", "low_confidence_locale_fallback"
            retrieval_policy = getattr(metadata, "retrieval_policy", None) if metadata else None
            allowed = getattr(retrieval_policy, "channels_allowed", None) if retrieval_policy else None
            if allowed:
                return "restrict_domains", "low_confidence_restrict_domains"
            return "refine_query", "low_confidence_refine_query"

        return "continue", "confidence_medium"

    def calculate_cost(self) -> int:
        # Tavily costs 1 credit per search, CSE is estimated from config
        cse_cost = int(getattr(self.config.runtime.search, "google_cse_cost", 0) or 0)
        # Cap CSE cost at 1 (same as Tavily search) to avoid cost explosion
        cse_cost = max(0, min(cse_cost, 1))

        return (self.tavily_calls + cse_cost * self.google_cse_calls)

    def can_afford(self, current_cost: int, max_cost: int | None) -> bool:
        if max_cost is None:
            return True
        return current_cost <= max_cost

    async def fetch_url_content(self, url: str, *, stage: int | None = None) -> str | None:
        """Securely fetch content via Tavily Extract (single URL, for back-compat)."""
        results = await self.fetch_urls_content_batch([url], stage=stage)
        content = results.get(url)
        if content:
            self.page_fetches += 1
        return content

    async def fetch_urls_content_batch(
        self,
        urls: list[str],
        *,
        stage: int | None = None,
    ) -> dict[str, str]:
        """
        Fetch content for multiple URLs using batch extraction.
        
        Uses WebSearchTool's internal cache and TavilyClient.extract_batch for efficiency.
        Tavily bills 1 credit per 5 URLs, so batching saves ~80% on extract costs.
        
        Returns:
            Dict mapping URL -> extracted content (only successful extractions)
        """
        if not urls:
            return {}
        
        url_map: dict[str, str] = {}
        missing: list[str] = []
        
        # Check cache first
        for u in urls:
            cached = self.web_tool._try_page_cache(u)
            if cached:
                url_map[u] = cached
            else:
                missing.append(u)
        
        Trace.event("search_mgr.batch_fetch.cache", {
            "hit": len(url_map),
            "miss": len(missing),
            "total": len(urls),
            "stage": stage,
        })
        
        if missing and self.web_tool.api_key:
            # Batch extract with 5-URL chunks
            BATCH_SIZE = 5
            batches = [missing[i:i + BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
            
            for batch_idx, batch in enumerate(batches):
                Trace.event("search_mgr.batch_fetch.call", {
                    "batch_index": batch_idx,
                    "urls_count": len(batch),
                    "stage": stage,
                })
                try:
                    data = await self.web_tool._tavily.extract_batch(urls=batch, format="markdown")
                    results = data.get("results", [])
                    
                    for item in results:
                        item_url = item.get("url")
                        if not item_url:
                            continue
                        raw = item.get("raw_content") or item.get("content") or ""
                        cleaned = self.web_tool._clean_extracted_text(raw)
                        if cleaned:
                            url_map[item_url] = cleaned
                            self.web_tool._write_page_cache(item_url, cleaned)
                            self.page_fetches += 1
                except Exception as e:
                    logger.warning("[SearchMgr] Batch extract failed: %s", e)
                    Trace.event("search_mgr.batch_fetch.error", {
                        "batch_index": batch_idx,
                        "urls_count": len(batch),
                        "error": str(e)[:200],
                        "stage": stage,
                    })
        
        return url_map

    async def apply_evidence_acquisition_ladder(
        self,
        sources: list[dict],
        *,
        max_fetches: int = 2,
        budget_context: str = "claim",
        claim_id: str | None = None,
        cache_only: bool = False,
    ) -> list[dict]:
        """
        EAL: Enrich snippet-only sources with content + quote candidates.
        
        Two-phase approach:
        1. If source has content but no quote, extract quote from existing content
        2. If source has no content, fetch URL and extract quote
        
        Args:
            sources: List of source dicts to enrich
            max_fetches: Maximum number of URL fetches
            budget_context: Which budget tracker to use:
                - "claim": For claim-specific evidence acquisition (default)
                - "inline": For inline source verification (separate budget)
            claim_id: If provided, use per-claim budget tracker (for deep mode).
                Each claim gets its own independent budget.
            cache_only: If True, only use cached content (no API calls).
                Use when orchestration layer has already pre-fetched URLs.
        
        Uses batch quote extraction for efficiency (single embedding call).
        """
        # Select appropriate budget tracker based on context
        if budget_context == "inline":
            tracker = self.inline_budget_tracker
        else:
            # Per-claim tracker if claim_id provided, else default
            tracker = self.get_claim_budget_tracker(claim_id)

        if not needs_evidence_acquisition_ladder(sources):
            return sources

        # Prefer higher relevance scores.
        candidates = sorted(
            [s for s in sources if isinstance(s, dict)],
            key=lambda s: float(s.get("relevance_score", 0.0) or 0.0),
            reverse=True,
        )

        # Phase 1: Collect URLs that need fetching (respecting budget and max_fetches)
        urls_to_fetch: list[str] = []
        src_by_url: dict[str, list[dict]] = {}  # URL -> sources that need this URL
        processed_urls: set[str] = set()

        # Track which sources have already been accounted in BudgetState.total_sources
        # to avoid double-counting (e.g., fetched then quoted).

        for src in candidates:
            if src.get("quote"):
                continue

            current_content = src.get("content") or src.get("snippet") or ""

            # Check if we can afford more fetches
            if len(urls_to_fetch) >= max_fetches:
                break

            # Check Bayesian budget allocation (dynamic limit)
            should_continue, reason = tracker.should_extract()
            if not should_continue:
                Trace.event(
                    "eal.budget_stop",
                    {
                        "reason": reason,
                        "budget_context": budget_context,
                        "claim_id": claim_id,
                        "budget_state": tracker.to_dict(),
                    },
                )
                break

            # Decide if this source needs fetching
            should_fetch = False
            if src.get("fulltext"):
                should_fetch = False  # Already has fulltext
            elif len(current_content) < 2000:
                should_fetch = True  # Short content, likely snippet

            if should_fetch:
                url = src.get("url") or src.get("link")
                if url and url not in processed_urls:
                    urls_to_fetch.append(url)
                    processed_urls.add(url)
                    # Track which sources need this URL
                    if url not in src_by_url:
                        src_by_url[url] = []
                    src_by_url[url].append(src)

        # Phase 1b: Fetch content for collected URLs
        # In cache_only mode, only read from cache (no API calls)
        fetch_count = 0
        if urls_to_fetch:
            if cache_only:
                # Cache-only mode: read from web_tool's cache without API calls
                fetched_map: dict[str, str] = {}
                for url in urls_to_fetch:
                    cached = self.web_tool._try_page_cache(url)
                    if cached:
                        fetched_map[url] = cached
                Trace.event("eal.cache_only_fetch", {
                    "urls_requested": len(urls_to_fetch),
                    "cache_hits": len(fetched_map),
                    "cache_misses": len(urls_to_fetch) - len(fetched_map),
                })
            else:
                # Normal mode: batch fetch with API calls
                fetched_map = await self.fetch_urls_content_batch(urls_to_fetch)

            # Apply fetched content to sources
            for url, content in fetched_map.items():
                for src in src_by_url.get(url, []):
                    src["content"] = content
                    src["fulltext"] = True
                    fetch_count += 1

                    # Record extract with Bayesian update
                    relevance = float(src.get("relevance_score", 0.5) or 0.5)
                    is_authoritative = src.get("source_tier") in ("A", "B")
                    tracker.record_extract(
                        relevance_score=relevance,
                        has_quote=False,  # Updated after quote extraction
                        is_authoritative=is_authoritative,
                    )
                    src["_budget_observed"] = True

        # Phase 1c: Collect sources that need quote extraction
        sources_for_quote_extraction: list[tuple[int, dict]] = []
        for idx, src in enumerate(candidates):
            current_content = src.get("content") or src.get("snippet") or ""
            if current_content and len(current_content) >= 50:
                claim_text = src.get("claim_text") or ""
                if claim_text:
                    sources_for_quote_extraction.append((idx, src))

        # Phase 2: Batch quote extraction (single embedding call for all sources)
        quoted_count = 0
        enriched_count = 0

        if sources_for_quote_extraction:
            try:
                from spectrue_core.utils.embedding_service import (
                    extract_best_quotes_batch_async,
                    EmbedService,
                )
                if EmbedService.is_available():
                    # Prepare batch items
                    batch_items = [
                        (src.get("claim_text", ""), src.get("content") or src.get("snippet") or "")
                        for _, src in sources_for_quote_extraction
                    ]

                    # Single async batch call
                    quotes = await extract_best_quotes_batch_async(batch_items)

                    # Assign quotes to sources
                    for i, (_, src) in enumerate(sources_for_quote_extraction):
                        best_quote = quotes[i] if i < len(quotes) else None

                        if best_quote:
                            src["quote"] = best_quote
                            src["quote_method"] = "semantic_batch"
                            src["eal_enriched"] = True
                            enriched_count += 1
                            quoted_count += 1
                            # If we did NOT fetch (cache_only miss / snippet-only),
                            # the tracker never saw this source, so total_sources stays 0.
                            # That breaks downstream evidence stats and explainability A.
                            relevance = float(src.get("relevance_score", 0.5) or 0.5)
                            is_authoritative = src.get("source_tier") in ("A", "B")
                            if not src.get("_budget_observed"):
                                tracker.state.update_from_source(
                                    relevance_score=relevance,
                                    has_quote=True,
                                    is_authoritative=is_authoritative,
                                )
                                src["_budget_observed"] = True
                            else:
                                # We already counted this source (via fetch). Only mark quote.
                                tracker.state.quotes_found += 1
                                tracker.state.alpha += 0.5  # Bonus for quote
                        else:
                            # Fallback to heuristic extraction
                            content = src.get("content") or src.get("snippet") or ""
                            heuristic_quotes = extract_quote_candidates(content)
                            if heuristic_quotes:
                                src["quote"] = heuristic_quotes[0]
                                src["quote_method"] = "heuristic"
                                src["eal_enriched"] = True
                                enriched_count += 1
                                quoted_count += 1
                                relevance = float(src.get("relevance_score", 0.5) or 0.5)
                                is_authoritative = src.get("source_tier") in ("A", "B")
                                if not src.get("_budget_observed"):
                                    tracker.state.update_from_source(
                                        relevance_score=relevance,
                                        has_quote=True,
                                        is_authoritative=is_authoritative,
                                    )
                                    src["_budget_observed"] = True
                                else:
                                    tracker.state.quotes_found += 1
                else:
                    # Embeddings unavailable - use heuristic for all
                    for _, src in sources_for_quote_extraction:
                        content = src.get("content") or src.get("snippet") or ""
                        heuristic_quotes = extract_quote_candidates(content)
                        if heuristic_quotes:
                            src["quote"] = heuristic_quotes[0]
                            src["quote_method"] = "heuristic"
                            src["eal_enriched"] = True
                            enriched_count += 1
                            quoted_count += 1
                            relevance = float(src.get("relevance_score", 0.5) or 0.5)
                            is_authoritative = src.get("source_tier") in ("A", "B")
                            if not src.get("_budget_observed"):
                                tracker.state.update_from_source(
                                    relevance_score=relevance,
                                    has_quote=True,
                                    is_authoritative=is_authoritative,
                                )
                                src["_budget_observed"] = True
                            else:
                                tracker.state.quotes_found += 1
            except ImportError:
                # Fallback to heuristic extraction
                for _, src in sources_for_quote_extraction:
                    content = src.get("content") or src.get("snippet") or ""
                    heuristic_quotes = extract_quote_candidates(content)
                    if heuristic_quotes:
                        src["quote"] = heuristic_quotes[0]
                        src["quote_method"] = "heuristic"
                        src["eal_enriched"] = True
                        enriched_count += 1
                        quoted_count += 1
                        relevance = float(src.get("relevance_score", 0.5) or 0.5)
                        is_authoritative = src.get("source_tier") in ("A", "B")
                        if not src.get("_budget_observed"):
                            tracker.state.update_from_source(
                                relevance_score=relevance,
                                has_quote=True,
                                is_authoritative=is_authoritative,
                            )
                            src["_budget_observed"] = True
                        else:
                            tracker.state.quotes_found += 1

        # ------------------------------------------------------------------
        # Reconcile evidence counters AFTER enrichment.
        # This keeps BudgetState.total_sources/relevant_sources consistent
        # even when fetch_count == 0 (cache_only/snippet-only paths).
        # ------------------------------------------------------------------
        from spectrue_core.verification.evidence.evidence_stats_reconcile import (
            reconcile_budget_state_from_sources,
        )
        reconcile_budget_state_from_sources(
            tracker,
            candidates,
            context=budget_context,
            claim_id=claim_id,
        )

        # Log budget state at end of EAL
        Trace.event(
            "evidence.ladder.summary",
            {
                "sources_in": len(sources or []),
                "candidates": len(candidates),
                "fetches": fetch_count,
                "quotes_added": quoted_count,
                "enriched": enriched_count,
                "batch_size": len(sources_for_quote_extraction),
                "budget_context": budget_context,
                "claim_id": claim_id,
                "budget_state": tracker.to_dict(),
            },
        )

        return sources

    async def check_oracle(self, query: str) -> dict | None:
        """Check Google Fact Check API (legacy method)."""
        self.oracle_calls += 1
        return await self.oracle_tool.search(query)

    async def check_oracle_hybrid(self, user_claim: str, intent: str = "news") -> OracleCheckResult:
        """
        Check Oracle with LLM semantic validation.
        
        Returns OracleCheckResult with status, relevance_score, is_jackpot.
        """
        self.oracle_calls += 1
        return await self.oracle_tool.search_and_validate(user_claim, intent)

    def _extract_domain(self, url: str) -> str:
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc.lower().replace("www.", "")
        except Exception:
            return ""

    def _filter_search_results(self, results: list[dict], intent: str) -> list[dict]:
        # Use reranking if policy_profile is available, otherwise fall back to filter
        if self.policy_profile:
            rerank_lambda = self.policy_profile.quality_thresholds.rerank_lambda
            # Rerank returns all results sorted by combined score
            # Apply top_k based on max_results from profile
            top_k = self.policy_profile.max_results or None
            return rerank_search_results(
                results,
                rerank_lambda=rerank_lambda,
                top_k=top_k,
                skip_extensions=SKIP_EXTENSIONS,
            )

        # Legacy fallback: hard filter by min_relevance
        min_relevance_score = 0.15
        return filter_search_results(
            results,
            min_relevance_score=min_relevance_score,
            skip_extensions=SKIP_EXTENSIONS,
        )

    async def search_unified(
        self,
        query: str,
        topic: str = "general",
        intent: str = "news",
        article_intent: str = "news",
        num_results: int = 5,  # Allow caller to control, default 5 for back-compat
    ) -> SearchResponse:
        """
        Unified search replacing Tier 1/2 split.
        
        DEPRECATED: Prefer search_phase() for new code. This method has legacy
        fallback ladder logic that may not be needed.
        
        Added Fallback Ladder:
        - If article_intent == "evergreen", force topic="general"
        - If topic="news" AND results are poor (<2 or low score), auto-retry with "general"
        """
        # Force general search for evergreen content
        if article_intent == "evergreen" and topic == "news":
            logger.debug("[SearchMgr] forcing topic='general' for evergreen intent (was 'news')")
            topic = "general"

        self.tavily_calls += 1
        # Use caller-provided num_results instead of hardcoded 5
        context, results = await self.web_tool.search(
            query,
            num_results=num_results,
            depth="advanced",
            topic=topic,
            skip_enrichment=True,  # Centralized enrichment at orchestration level
        )

        filtered = self._filter_search_results(results, intent)
        should_fallback, fallback_reason, max_score = should_fallback_news_to_general(topic, filtered)

        if should_fallback:
            logger.debug(f"[SearchMgr] Fallback triggered: {fallback_reason}. Retrying with topic='general'.")

            # Retry with "general"
            self.tavily_calls += 1
            fb_context, fb_results = await self.web_tool.search(
                query,
                num_results=num_results,  # Use same limit for fallback
                depth="advanced",
                topic="general",
                skip_enrichment=True,  # Centralized enrichment at orchestration level
            )
            fb_filtered = self._filter_search_results(fb_results, intent)

            if prefer_fallback_results(
                original_filtered=filtered,
                original_max_score=max_score,
                fallback_filtered=fb_filtered,
            ):
                fb_count = len(fb_filtered)
                fb_max_score = max([float(r.get("score", 0) or 0.0) for r in fb_filtered]) if fb_filtered else 0.0
                logger.debug(f"[SearchMgr] Fallback successful. Using general results (count={fb_count}, max={fb_max_score:.2f})")
                filtered = fb_filtered
                # Reconstruct context from the winner
                # Note: We reconstruct `context` below based on `filtered`, so just updating `filtered` is enough.
            else:
                logger.debug("[SearchMgr] Fallback yielded no improvement. Keeping original results.")

        new_context = build_context_from_sources(filtered)
        return new_context, filtered

    async def search_phase(
        self,
        query: str,
        *,
        topic: str = "general",
        depth: str = "basic",
        max_results: int = 5,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse:
        """
        PhaseRunner search primitive.
        
        Unlike `search_unified` (legacy pipeline with topic fallback ladder and
        additional filtering for older Tavily response shapes), this is a thin
        wrapper around `WebSearchTool.search()` that:
        - respects `depth` and `max_results`
        - supports `include_domains`/`exclude_domains`
        - returns the canonical tuple `(context, sources)`
        """
        self.tavily_calls += 1
        return await self.web_tool.search(
            query,
            num_results=max_results,
            depth=depth,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            topic=topic,
            skip_enrichment=True,  # Centralized enrichment at orchestration level
        )

    async def search_tier1(self, query: str, domains: list[str]) -> SearchResponse:
        """Perform Tier 1 search on trusted domains (Legacy/Fallback)."""
        self.tavily_calls += 1
        return await self.web_tool.search(
            query, 
            depth="advanced", 
            include_domains=domains,
            skip_enrichment=True,  # Centralized enrichment at orchestration level
        )

    async def search_tier2(
        self,
        query: str,
        exclude_domains: list[str] = None,
        topic: str = "general",
    ) -> SearchResponse:
        """Perform Tier 2 (General) search (Legacy/Fallback)."""
        self.tavily_calls += 1
        return await self.web_tool.search(
            query,
            depth="advanced",
            exclude_domains=exclude_domains,
            topic=topic,
            skip_enrichment=True,  # Centralized enrichment at orchestration level
        )

    async def search_google_cse(self, query: str, lang: str) -> list[dict]:
        """Perform Google CSE search."""
        self.google_cse_calls += 1
        return await self.cse_tool.search(query, lang=lang)

    def get_search_meta(self) -> dict:
        """Return usage metrics for current run."""
        return {
            "tavily_calls": self.tavily_calls,
            "google_cse_calls": self.google_cse_calls,
            "page_fetches": self.page_fetches,
            "oracle_calls": self.oracle_calls,
        }
