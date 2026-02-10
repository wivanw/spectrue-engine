from __future__ import annotations

from typing import Protocol, Any, TYPE_CHECKING
from spectrue_core.domain.evidence.model import OracleCheckResult

if TYPE_CHECKING:
    from spectrue_core.domain.claims.model import Claim
    from spectrue_core.domain.evidence.model import ArticleIntent


class FactCheckerAgentProtocol(Protocol):
    """
    Protocol for the fact-checker agent.
    
    This allows use_cases to depend on the agent without importing the implementation
    from the agents layer.
    """
    oracle_skill: Any
    llm_client: Any
    edge_typing_skill: Any

    async def extract_claims(
        self, text: str, *, lang: str = "en", max_claims: int = 20, anchors: list | None = None
    ) -> tuple[list[Any], bool, Any, str]:
        ...

    async def score_evidence(self, pack: Any, *, lang: str = "en") -> dict:
        ...

    async def generate_search_queries(self, fact: str, context: str = "", lang: str = "en", content_lang: str | None = None) -> list[str]:
        ...
