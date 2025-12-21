from spectrue_core.tools.search_tool import WebSearchTool
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool
from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.evidence_pack import OracleCheckResult
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
        
    def reset_metrics(self):
        self.tavily_calls = 0
        self.google_cse_calls = 0
        self.page_fetches = 0
        self.oracle_calls = 0

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
        """M65: Filter search results (Score > 0.15, No .txt/.xml)."""
        out = []
        for r in results:
            # 1. Score Cutoff (M65 Goal 2)
            # Tavily returns relevance_score (0..1), sometimes missing
            score = r.get("relevance_score")
            # If score is present and valid float, check it.
            if isinstance(score, (int, float)) and score < 0.15:
                continue
            
            # 2. File Extension
            url_str = r.get("link", "") or r.get("url", "")
            if url_str.lower().endswith(SKIP_EXTENSIONS):
                continue
            
            # NOTE: We delegate domain quality filtering to Tavily's topic="news" 
            # and the score cutoff. Hardcoded JUNK_DOMAINS are removed (Technical Debt).
            
            out.append(r)
        return out

    async def search_unified(self, query: str, topic: str = "general", intent: str = "news", article_intent: str = "news") -> tuple[str, list[dict]]:
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
        
        # M76: Fallback Logic (News -> General)
        # Condition: We tried "news", but got < 2 valid results OR best score is very low (< 0.2)
        should_fallback = False
        fallback_reason = ""
        
        if topic == "news":
            valid_count = len(filtered)
            max_score = max([float(r.get("score", 0)) for r in filtered]) if filtered else 0.0
            
            if valid_count < 2:
                should_fallback = True
                fallback_reason = f"few_results ({valid_count})"
            elif max_score < 0.2:
                should_fallback = True
                fallback_reason = f"low_relevance ({max_score:.2f})"
        
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
            
            # Prefer fallback results if they are better
            fb_count = len(fb_filtered)
            fb_max_score = max([float(r.get("score", 0)) for r in fb_filtered]) if fb_filtered else 0.0
            
            # Use fallback if it produced ANY results and the original was practically empty,
            # OR if fallback has significantly better relevance.
            if fb_count > 0 and (len(filtered) == 0 or fb_max_score > max_score):
                logger.info(f"[SearchMgr] Fallback successful. Using general results (count={fb_count}, max={fb_max_score:.2f})")
                filtered = fb_filtered
                # Reconstruct context from the winner
                # Note: We reconstruct `context` below based on `filtered`, so just updating `filtered` is enough.
            else:
                logger.info("[SearchMgr] Fallback yielded no improvement. Keeping original results.")

        # Reconstruct context derived from filtered sources only
        def format_source(obj: dict) -> str:
            return f"Source: {obj.get('title')}\nURL: {obj.get('link')}\nContent: {obj.get('snippet')}\n---"
            
        new_context = "\n".join([format_source(obj) for obj in filtered])
        
        return new_context, filtered

    async def search_tier1(self, query: str, domains: list[str]) -> tuple[str, list[dict]]:
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
        topic: str = "general"
    ) -> tuple[str, list[dict]]:
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
