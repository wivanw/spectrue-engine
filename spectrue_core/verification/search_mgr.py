from spectrue_core.tools.search_tool import WebSearchTool
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool
from spectrue_core.config import SpectrueConfig
from spectrue_core.utils.trace import Trace
import logging

logger = logging.getLogger(__name__)

SEARCH_COSTS = {"basic": 80, "advanced": 160}
MODEL_COSTS = {"gpt-5-nano": 5, "gpt-5-mini": 20, "gpt-5.2": 100}

class SearchManager:
    """
    Manages search tools, execution, and cost budgeting.
    """
    def __init__(self, config: SpectrueConfig):
        self.config = config
        self.web_tool = WebSearchTool(config)
        self.oracle_tool = GoogleFactCheckTool(config)
        self.cse_tool = GoogleCSESearchTool(config)
        
        # Metrics state for current run
        self.tavily_calls = 0
        self.google_cse_calls = 0
        self.page_fetches = 0
        
    def reset_metrics(self):
        self.tavily_calls = 0
        self.google_cse_calls = 0
        self.page_fetches = 0

    def calculate_cost(self, model: str, search_type: str) -> int:
        """Calculate total billed cost based on operations performed."""
        model_cost = int(MODEL_COSTS.get(model, 20))
        per_search_cost = int(SEARCH_COSTS.get(search_type, 80))
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

    async def check_oracle(self, query: str, lang: str) -> dict | None:
        """Check Google Fact Check API."""
        return await self.oracle_tool.search(query, lang)

    async def search_tier1(self, query: str, domains: list[str]) -> tuple[str, list[dict]]:
        """Perform Tier 1 search on trusted domains."""
        self.tavily_calls += 1
        return await self.web_tool.search(
            query, 
            search_depth="advanced", 
            domains=domains
        )

    async def search_tier2(self, query: str, exclude_domains: list[str] = None) -> tuple[str, list[dict]]:
        """Perform Tier 2 (General) search."""
        self.tavily_calls += 1
        return await self.web_tool.search(
            query,
            search_depth="advanced",
            exclude_domains=exclude_domains
        )
    
    async def search_google_cse(self, query: str, lang: str) -> list[dict]:
        """Perform Google CSE search."""
        self.google_cse_calls += 1
        return await self.cse_tool.search(query, lang=lang)

    def get_search_meta(self) -> dict:
        return {
            "tavily_calls": self.tavily_calls,
            "google_cse_calls": self.google_cse_calls,
            "page_fetches": self.page_fetches,
        }
