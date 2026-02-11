"""
Models and types for evidence spillover.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

NormalizeUrl = Callable[[str], str]
SlotsFromAssertionKey = Callable[[str], set[str]]
RequiredSlotsForTarget = Callable[[str], set[str]]
MergeCovers = Callable[[Any, set[str]], set[str]]
ClaimEventSignature = Callable[[dict[str, Any]], Any]
EvidenceEventSignature = Callable[[dict[str, Any]], Any]
SignatureCompatible = Callable[[Any, Any], bool]


@dataclass(frozen=True)
class SpilloverChoice:
    claim_id: str
    cluster_id: str
    count: int
    urls: list[str]
    topic_boost_used: int


@dataclass(frozen=True)
class SpilloverResult:
    combined_sources: list[dict[str, Any]]
    evidence_by_claim: dict[str, list[dict[str, Any]]]
    transferred_items: list[dict[str, Any]]
    transferred_total: int
    touched_claims: int
    rejections: dict[str, int]
    choices: list[SpilloverChoice]
