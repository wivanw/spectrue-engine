from spectrue_core.tools.search_tool import WebSearchTool
from spectrue_core.tools.google_fact_check import GoogleFactCheckTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool
from spectrue_core.config import SpectrueConfig
from spectrue_core.verification.evidence_pack import OracleCheckResult
import logging

logger = logging.getLogger(__name__)

SEARCH_COSTS = {"smart": 100}
MODEL_COSTS = {"gpt-5-nano": 5, "gpt-5-mini": 20, "gpt-5.2": 100}

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
        
        Returns OracleCheckResult with relevance_score for three-scenario flow:
        - JACKPOT (>0.9): Stop pipeline, return Oracle result
        - EVIDENCE (0.5-0.9): Add to evidence pack, continue search
        - MISS (<0.5): Ignore, proceed to standard search
        
        Args:
            user_claim: The claim to search for
            intent: Article intent (news/evergreen/official/opinion/prediction)
            
        Returns:
            OracleCheckResult with status, relevance_score, is_jackpot
        """
        self.oracle_calls += 1
        return await self.oracle_tool.search_and_validate(user_claim, intent)


    async def search_tier1(self, query: str, domains: list[str]) -> tuple[str, list[dict]]:
        """Perform Tier 1 search on trusted domains."""
        self.tavily_calls += 1
        return await self.web_tool.search(
            query, 
            depth="advanced", 
            include_domains=domains
        )

    async def search_tier2(self, query: str, exclude_domains: list[str] = None) -> tuple[str, list[dict]]:
        """Perform Tier 2 (General) search."""
        self.tavily_calls += 1
        return await self.web_tool.search(
            query,
            depth="advanced",
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
