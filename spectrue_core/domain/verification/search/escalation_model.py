# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EscalationConfig:
    """Configuration for escalation thresholds."""
    min_relevance_threshold: float = 0.2
    min_usable_snippets: int = 2
    max_query_length: int = 80
    min_snippet_chars: int = 50
    max_token_length: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_relevance_threshold": self.min_relevance_threshold,
            "min_usable_snippets": self.min_usable_snippets,
            "max_query_length": self.max_query_length,
            "min_snippet_chars": self.min_snippet_chars,
            "max_token_length": self.max_token_length,
        }


@dataclass
class QueryVariant:
    """A deterministic query variant built from claim fields."""
    query_id: str
    text: str
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "strategy": self.strategy,
        }


@dataclass
class RetrievalOutcome:
    """Observable quality signals from RAW search results."""
    sources_count: int
    best_relevance: float
    usable_snippets_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_count": self.sources_count,
            "best_relevance": self.best_relevance,
            "usable_snippets_count": self.usable_snippets_count,
        }


@dataclass
class EscalationPass:
    """Configuration for one escalation pass."""
    pass_id: str
    search_depth: str
    max_results: int
    topic: str | None
    include_domains_relaxed: bool
    query_ids: list[str]
    trigger_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "topic": self.topic,
            "include_domains_relaxed": self.include_domains_relaxed,
            "query_ids": self.query_ids,
        }
