from spectrue_core.tools.search_tool import WebSearchTool
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool
from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.evidence_pack import OracleCheckResult
from spectrue_core.verification.evidence import (
    needs_evidence_acquisition_ladder,
    extract_quote_candidates,
)
from spectrue_core.verification.types import SearchResponse
from spectrue_core.verification.search_policy import (
    build_context_from_sources,
    filter_search_results,
    prefer_fallback_results,
    should_fallback_news_to_general,
    SearchPolicyProfile,
)
from spectrue_core.utils.trace import Trace
import logging

logger = logging.getLogger(__name__)

SEARCH_COSTS = {"smart": 100}
MODEL_COSTS = {"gpt-5-nano": 5, "gpt-5-mini": 20, "gpt-5.2": 100}

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
            oracle_validator: Optional OracleValidationSkill for M63 hybrid mode
        """
        self.config = config
        self.web_tool = WebSearchTool(config)
        # M63: Pass validator to GoogleFactCheckTool for hybrid mode
        self.oracle_tool = GoogleFactCheckTool(config, oracle_validator=oracle_validator)
        self.cse_tool = GoogleCSESearchTool(config)
        
        # Metrics state for current run
        self.tavily_calls = 0
        self.google_cse_calls = 0
        self.page_fetches = 0
        self.oracle_calls = 0  # M63: Track Oracle API calls
        self.global_extracts_used = 0  # M108: Global EAL limit
        self.policy_profile: SearchPolicyProfile | None = None
        
    def reset_metrics(self):
        self.tavily_calls = 0
        self.google_cse_calls = 0
        self.page_fetches = 0
        self.oracle_calls = 0
        self.global_extracts_used = 0  # M108: Global EAL limit
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

    def estimate_hop_cost(self, *, search_type: str | None = None) -> float:
        search_key = (search_type or "smart").lower()
        return float(SEARCH_COSTS.get(search_key, 100))

    def calculate_cost(self, model: str, search_type: str) -> int:
        """Calculate total billed cost based on operations performed."""
        model_cost = int(MODEL_COSTS.get(model, 20))
        per_search_cost = int(SEARCH_COSTS.get(search_type, 100))
        cse_cost = int(getattr(self.config.runtime.search, "google_cse_cost", 0) or 0)
        # Cap CSE cost to avoiding surpassing search cost
        cse_cost = max(0, min(cse_cost, per_search_cost))
        
        return (
            model_cost + 
            (per_search_cost * self.tavily_calls) + 
            (cse_cost * self.google_cse_calls)
        )

    def can_afford(self, current_cost: int, max_cost: int | None) -> bool:
        if max_cost is None:
            return True
        return current_cost <= max_cost

    async def fetch_url_content(self, url: str) -> str | None:
        """Securely fetch content via Tavily Extract."""
        content = await self.web_tool._fetch_extract_text(url)
        if content:
            self.page_fetches += 1
        return content

    async def apply_evidence_acquisition_ladder(
        self,
        sources: list[dict],
        *,
        max_fetches: int = 2,
    ) -> list[dict]:
        """
        EAL: Enrich snippet-only sources with content + quote candidates.
        
        M104: Two-phase approach:
        1. If source has content but no quote, extract quote from existing content
        2. If source has no content, fetch URL and extract quote
        
        M111: Uses batch quote extraction for efficiency (single embedding call).
        """
        if not needs_evidence_acquisition_ladder(sources):
            return sources

        fetch_count = 0
        # Prefer higher relevance scores.
        candidates = sorted(
            [s for s in sources if isinstance(s, dict)],
            key=lambda s: float(s.get("relevance_score", 0.0) or 0.0),
            reverse=True,
        )

        processed_urls = set()
        
        # Phase 1: Fetch content for sources that need it
        sources_for_quote_extraction: list[tuple[int, dict]] = []  # (index, source)
        
        for idx, src in enumerate(candidates):
            if src.get("quote"):
                continue

            current_content = src.get("content") or src.get("snippet") or ""
            should_fetch = False
            
            if fetch_count < max_fetches:
                # M108: Check global limit (across all claims)
                if self.global_extracts_used >= 4:
                    Trace.event(
                        "eal.global_limit_reached",
                        {"global_extracts_used": self.global_extracts_used, "max": 4},
                    )
                    break
                # If explicitly marked as fulltext, don't re-fetch
                if src.get("fulltext"):
                    should_fetch = False
                # If content is very short (likely snippet) or missing, fetch
                elif len(current_content) < 2000:
                    should_fetch = True
            
            if should_fetch:
                url = src.get("url") or src.get("link")
                if url:
                    # M105: Deduplication to avoid wasted fetches
                    if url in processed_urls:
                        continue
                    
                    fetched = await self.fetch_url_content(url)
                    processed_urls.add(url)
                    
                    if fetched:
                        src["content"] = fetched
                        src["fulltext"] = True
                        current_content = fetched
                        fetch_count += 1
                        self.global_extracts_used += 1  # M108: Track globally
            
            # Collect sources that need quote extraction
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

        Trace.event(
            "evidence.ladder.summary",
            {
                "sources_in": len(sources or []),
                "candidates": len(candidates),
                "fetches": fetch_count,
                "quotes_added": quoted_count,
                "enriched": enriched_count,
                "batch_size": len(sources_for_quote_extraction),
            },
        )

        return sources

    async def check_oracle(self, query: str) -> dict | None:
        """Check Google Fact Check API (legacy method)."""
        self.oracle_calls += 1
        return await self.oracle_tool.search(query)

    async def check_oracle_hybrid(self, user_claim: str, intent: str = "news") -> OracleCheckResult:
        """
        M63: Check Oracle with LLM semantic validation.
        
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
        # Back-compat: older tests and callers reference this helper on SearchManager.
        # `intent` is currently unused (policy is score + extension based).
        min_relevance_score = 0.15
        if self.policy_profile:
            min_relevance_score = self.policy_profile.quality_thresholds.min_relevance_score
        return filter_search_results(
            results,
            min_relevance_score=min_relevance_score,
            skip_extensions=SKIP_EXTENSIONS,
        )

    async def search_unified(self, query: str, topic: str = "general", intent: str = "news", article_intent: str = "news") -> SearchResponse:
        """
        M65: Unified search replacing Tier 1/2 split.
        Performs single search (limit=5) and filters garbage.
        
        M76: Added Fallback Ladder:
        - If article_intent == "evergreen", force topic="general"
        - If topic="news" AND results are poor (<2 or low score), auto-retry with "general"
        """
        # M76: Force general search for evergreen content
        if article_intent == "evergreen" and topic == "news":
            logger.debug("[SearchMgr] forcing topic='general' for evergreen intent (was 'news')")
            topic = "general"

        self.tavily_calls += 1
        # Use limit=5 to strictly match Tavily billing unit and avoid noise from results 6-10
        context, results = await self.web_tool.search(
            query,
            num_results=5,
            depth="advanced",
            topic=topic
        )
        
        filtered = self._filter_search_results(results, intent)
        should_fallback, fallback_reason, max_score = should_fallback_news_to_general(topic, filtered)

        if should_fallback:
            logger.debug(f"[SearchMgr] Fallback triggered: {fallback_reason}. Retrying with topic='general'.")
            
            # Retry with "general"
            self.tavily_calls += 1
            fb_context, fb_results = await self.web_tool.search(
                query,
                num_results=5,
                depth="advanced",
                topic="general"
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
        M83: PhaseRunner search primitive.
        
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
        )

    async def search_tier1(self, query: str, domains: list[str]) -> SearchResponse:
        """Perform Tier 1 search on trusted domains (Legacy/Fallback)."""
        self.tavily_calls += 1
        return await self.web_tool.search(
            query, 
            depth="advanced", 
            include_domains=domains
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
            topic=topic
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
