# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

"""
Retrieval Sanity Gate

Deterministic off-topic detection for search results using structured anchor terms.
Ensures evidence is topically relevant to the claim before proceeding to audit/judge.

Key components:
- normalize_anchor_terms: Build normalized token set from claim structured fields
- check_sanity_gate: Check if sources match anchor terms (no lexical heuristics)
- SanityGateResult: Result dataclass with decision and diagnostics

Uses ONLY structured fields: subject_entities, context_entities, retrieval_seed_terms.
NO lexical heuristics like "if contains X".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from spectrue_core.utils.trace import Trace


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SanityGateConfig:
    """Configuration for sanity gate thresholds."""
    
    min_overlap_count: int = 1
    """Minimum number of anchor terms that must appear in sources to pass."""
    
    max_anchor_terms: int = 8
    """Maximum anchor terms to consider (for overlap ratio calculation)."""
    
    min_relevance_threshold: float = 0.2
    """Minimum Tavily relevance score to consider (combined with overlap)."""
    
    min_token_len: int = 3
    """Minimum token length to include in anchor terms."""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "min_overlap_count": self.min_overlap_count,
            "max_anchor_terms": self.max_anchor_terms,
            "min_relevance_threshold": self.min_relevance_threshold,
            "min_token_len": self.min_token_len,
        }


DEFAULT_SANITY_GATE_CONFIG = SanityGateConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Anchor Term Normalization
# ─────────────────────────────────────────────────────────────────────────────


# Regex to strip punctuation but keep unicode letters/digits
_PUNCT_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_token(token: str) -> str:
    """Normalize a token: lowercase, strip punctuation and whitespace."""
    normalized = _PUNCT_PATTERN.sub("", token.lower().strip())
    return normalized


def normalize_anchor_terms(
    subject_entities: list[str],
    context_entities: list[str],
    retrieval_seed_terms: list[str],
    config: SanityGateConfig | None = None,
) -> set[str]:
    """
    Build normalized anchor term set from claim structured fields.
    
    Normalization:
    - Lowercase
    - Strip punctuation
    - Drop tokens < min_token_len chars
    
    Args:
        subject_entities: Claim's subject entities
        context_entities: Inherited context entities
        retrieval_seed_terms: Seed terms for retrieval
        config: Optional config for filtering
    
    Returns:
        Set of normalized anchor terms (unique, lowercase, no punctuation)
    """
    cfg = config or DEFAULT_SANITY_GATE_CONFIG
    min_len = cfg.min_token_len
    
    all_terms: list[str] = []
    all_terms.extend(subject_entities or [])
    all_terms.extend(context_entities or [])
    all_terms.extend(retrieval_seed_terms or [])
    
    anchor_set: set[str] = set()
    for term in all_terms:
        if not isinstance(term, str):
            continue
        # Split multi-word terms into individual tokens
        for word in term.split():
            normalized = _normalize_token(word)
            if len(normalized) >= min_len:
                anchor_set.add(normalized)
    
    return anchor_set


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Gate Result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SanityGateResult:
    """Result of sanity gate check."""
    
    decision: Literal["pass", "off_topic"]
    """Gate decision: 'pass' = proceed to audit, 'off_topic' = escalate or fail."""
    
    max_overlap_count: int
    """Maximum overlap count across all sources."""
    
    overlap_ratio: float
    """Ratio of max_overlap to min(anchor_terms, K)."""
    
    anchor_terms_count: int
    """Number of anchor terms used for matching."""
    
    best_relevance: float
    """Best Tavily relevance score among sources."""
    
    reasons: list[str] = field(default_factory=list)
    """Diagnostic reason codes explaining the decision."""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "max_overlap_count": self.max_overlap_count,
            "overlap_ratio": round(self.overlap_ratio, 3),
            "anchor_terms_count": self.anchor_terms_count,
            "best_relevance": round(self.best_relevance, 3),
            "reasons": list(self.reasons),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Gate Check
# ─────────────────────────────────────────────────────────────────────────────


def _tokenize_text(text: str, min_token_len: int = 3) -> set[str]:
    """Tokenize text into normalized tokens."""
    tokens: set[str] = set()
    for word in text.split():
        normalized = _normalize_token(word)
        if len(normalized) >= min_token_len:
            tokens.add(normalized)
    return tokens


def _compute_overlap(
    anchor_terms: set[str],
    source_tokens: set[str],
) -> int:
    """Compute number of anchor terms present in source tokens."""
    return len(anchor_terms.intersection(source_tokens))


def check_sanity_gate(
    sources: list[dict[str, Any]],
    anchor_terms: set[str],
    config: SanityGateConfig | None = None,
) -> SanityGateResult:
    """
    Check if sources are topically relevant to the claim.
    
    Uses structural overlap counting on title + snippet text.
    NOT lexical heuristics - only checks presence of structured anchor terms.
    
    Args:
        sources: List of search result dicts (with title, snippet, score)
        anchor_terms: Normalized anchor terms from claim structured fields
        config: Optional config for thresholds
    
    Returns:
        SanityGateResult with decision and diagnostics
    """
    cfg = config or DEFAULT_SANITY_GATE_CONFIG
    min_token_len = cfg.min_token_len
    
    reasons: list[str] = []
    max_overlap = 0
    best_relevance = 0.0
    
    # Handle empty sources
    if not sources:
        return SanityGateResult(
            decision="off_topic",
            max_overlap_count=0,
            overlap_ratio=0.0,
            anchor_terms_count=len(anchor_terms),
            best_relevance=0.0,
            reasons=["no_sources"],
        )
    
    # Handle empty anchor terms (shouldn't happen but guard)
    if not anchor_terms:
        reasons.append("no_anchor_terms")
        # Can't determine relevance → conservative pass
        return SanityGateResult(
            decision="pass",
            max_overlap_count=0,
            overlap_ratio=0.0,
            anchor_terms_count=0,
            best_relevance=0.0,
            reasons=reasons,
        )
    
    # Check each source
    for src in sources:
        if not isinstance(src, dict):
            continue
        
        # Extract text to check
        title = src.get("title", "") or ""
        snippet = src.get("snippet", "") or src.get("content", "") or ""
        combined_text = f"{title} {snippet}"
        
        # Tokenize and compute overlap
        source_tokens = _tokenize_text(combined_text, min_token_len)
        overlap = _compute_overlap(anchor_terms, source_tokens)
        max_overlap = max(max_overlap, overlap)
        
        # Track best relevance
        score = src.get("score") or src.get("relevance_score") or 0.0
        if isinstance(score, (int, float)):
            best_relevance = max(best_relevance, float(score))
    
    # Compute overlap ratio
    k = min(len(anchor_terms), cfg.max_anchor_terms)
    overlap_ratio = max_overlap / k if k > 0 else 0.0
    
    # Decision logic
    # OFF_TOPIC if:
    # 1. No overlap at all AND relevance is low
    # 2. No overlap at all (even if relevance is high - Tavily can hallucinate relevance)
    if max_overlap < cfg.min_overlap_count:
        if best_relevance < cfg.min_relevance_threshold:
            reasons.append("no_overlap_low_relevance")
        else:
            reasons.append("no_overlap_despite_relevance")
        decision: Literal["pass", "off_topic"] = "off_topic"
    else:
        reasons.append("overlap_ok")
        decision = "pass"
    
    return SanityGateResult(
        decision=decision,
        max_overlap_count=max_overlap,
        overlap_ratio=overlap_ratio,
        anchor_terms_count=len(anchor_terms),
        best_relevance=best_relevance,
        reasons=reasons,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trace Events
# ─────────────────────────────────────────────────────────────────────────────


def trace_sanity_gate(
    claim_id: str,
    pass_id: str,
    result: SanityGateResult,
) -> None:
    """Log sanity gate trace event."""
    Trace.event(
        "search.sanity",
        {
            "claim_id": claim_id,
            "pass_id": pass_id,
            **result.to_dict(),
        },
    )
