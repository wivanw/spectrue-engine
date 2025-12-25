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
        self.policy_profile: SearchPolicyProfile | None = None
        
    def reset_metrics(self):
        self.tavily_calls = 0
        self.google_cse_calls = 0
        self.page_fetches = 0
        self.oracle_calls = 0
        self.policy_profile = None

    def set_policy_profile(self, profile: SearchPolicyProfile | None) -> None:
        self.policy_profile = profile

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

        for src in candidates:
            # M104: If already has quote, skip entirely
            if src.get("quote"):
                continue
            
            # M104: If has content but no quote, extract quote from existing content
            existing_content = src.get("content") or src.get("snippet")
            if existing_content and len(existing_content) >= 50:
                quotes = extract_quote_candidates(existing_content)
                if quotes:
                    src["quote"] = quotes[0]
                    src["eal_enriched"] = True
                continue  # Don't count as fetch, just extraction
            
            # If no content at all, fetch URL (up to max_fetches)
            if fetch_count >= max_fetches:
                continue
            url = src.get("url") or src.get("link")
            if not url:
                continue
            content = await self.fetch_url_content(url)
            if not content:
                continue
            src["content"] = content
            quotes = extract_quote_candidates(content)
            if quotes:
                src["quote"] = quotes[0]
            src["eal_enriched"] = True
            fetch_count += 1

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
            logger.info("[SearchMgr] forcing topic='general' for evergreen intent (was 'news')")
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
            logger.info(f"[SearchMgr] Fallback triggered: {fallback_reason}. Retrying with topic='general'.")
            
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
                logger.info(f"[SearchMgr] Fallback successful. Using general results (count={fb_count}, max={fb_max_score:.2f})")
                filtered = fb_filtered
                # Reconstruct context from the winner
                # Note: We reconstruct `context` below based on `filtered`, so just updating `filtered` is enough.
            else:
                logger.info("[SearchMgr] Fallback yielded no improvement. Keeping original results.")

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
