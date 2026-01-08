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
Coverage Skeleton Extraction

Phase 1 of 2-phase claim extraction:
1. Skeleton: Extract all events/measurements/quotes/policies with raw_span
2. Claims: Convert skeleton items to atomic verifiable claims

This module provides:
- Skeleton dataclasses for structured extraction
- Coverage analyzers (regex-based, language-agnostic)
- Coverage validation with trace warnings
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.utils.trace import Trace


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SkeletonEvent:
    """An event extracted from text (something happened)."""
    
    id: str
    """Unique identifier within skeleton (e.g., 'evt_1')."""
    
    subject_entities: list[str]
    """Who/what is involved in the event."""
    
    verb_phrase: str
    """What happened (action/state change)."""
    
    time_anchor: dict | None
    """Time reference if present: {type, value}."""
    
    location_anchor: str | None
    """Location if mentioned."""
    
    raw_span: str
    """Exact substring from input text."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_entities": self.subject_entities,
            "verb_phrase": self.verb_phrase,
            "time_anchor": self.time_anchor,
            "location_anchor": self.location_anchor,
            "raw_span": self.raw_span,
        }


@dataclass
class QuantityMention:
    """A numeric value with optional unit."""
    
    value: str
    """The numeric value as string."""
    
    unit: str | None
    """Unit if present (%, $, million, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "unit": self.unit}


@dataclass
class SkeletonMeasurement:
    """A measurement/statistic extracted from text."""
    
    id: str
    """Unique identifier within skeleton (e.g., 'msr_1')."""
    
    subject_entities: list[str]
    """What is being measured."""
    
    metric: str
    """What metric (revenue, growth, count, etc.)."""
    
    quantity_mentions: list[QuantityMention]
    """Numeric values with units."""
    
    time_anchor: dict | None
    """Time reference if present."""
    
    raw_span: str
    """Exact substring from input text."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_entities": self.subject_entities,
            "metric": self.metric,
            "quantity_mentions": [q.to_dict() for q in self.quantity_mentions],
            "time_anchor": self.time_anchor,
            "raw_span": self.raw_span,
        }


@dataclass
class SkeletonQuote:
    """A quote/statement extracted from text."""
    
    id: str
    """Unique identifier within skeleton (e.g., 'qot_1')."""
    
    speaker_entities: list[str]
    """Who said it."""
    
    quote_text: str
    """What was said (may be paraphrased)."""
    
    raw_span: str
    """Exact substring from input text."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "speaker_entities": self.speaker_entities,
            "quote_text": self.quote_text,
            "raw_span": self.raw_span,
        }


@dataclass
class SkeletonPolicy:
    """A policy/regulation/rule extracted from text."""
    
    id: str
    """Unique identifier within skeleton (e.g., 'pol_1')."""
    
    subject_entities: list[str]
    """Who issued or is affected by the policy."""
    
    policy_action: str
    """What the policy does/requires."""
    
    time_anchor: dict | None
    """Time reference if present."""
    
    raw_span: str
    """Exact substring from input text."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_entities": self.subject_entities,
            "policy_action": self.policy_action,
            "time_anchor": self.time_anchor,
            "raw_span": self.raw_span,
        }


@dataclass
class CoverageSkeleton:
    """Complete skeleton extracted from text."""
    
    events: list[SkeletonEvent] = field(default_factory=list)
    measurements: list[SkeletonMeasurement] = field(default_factory=list)
    quotes: list[SkeletonQuote] = field(default_factory=list)
    policies: list[SkeletonPolicy] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [e.to_dict() for e in self.events],
            "measurements": [m.to_dict() for m in self.measurements],
            "quotes": [q.to_dict() for q in self.quotes],
            "policies": [p.to_dict() for p in self.policies],
        }

    def total_items(self) -> int:
        return len(self.events) + len(self.measurements) + len(self.quotes) + len(self.policies)

    def counts(self) -> dict[str, int]:
        return {
            "events": len(self.events),
            "measurements": len(self.measurements),
            "quotes": len(self.quotes),
            "policies": len(self.policies),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Coverage Analyzers (regex-based, language-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

# Patterns for time mentions
TIME_PATTERNS = [
    # YYYY-MM-DD
    r"\b\d{4}-\d{2}-\d{2}\b",
    # YYYY/MM/DD
    r"\b\d{4}/\d{2}/\d{2}\b",
    # YYYY-MM
    r"\b\d{4}-\d{2}\b",
    # YYYY alone (but not in middle of number)
    r"(?<!\d)\b(19|20)\d{2}\b(?!\d)",
    # Month YYYY (English)
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    # Mon YYYY
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.,]?\s+\d{4}\b",
    # Q1/Q2/Q3/Q4 YYYY
    r"\bQ[1-4]\s*\d{4}\b",
]

# Combined pattern
_TIME_REGEX = re.compile("|".join(f"({p})" for p in TIME_PATTERNS), re.IGNORECASE)


def extract_time_mentions_count(text: str) -> int:
    """
    Count time mentions in text using regex patterns.
    
    Detects: YYYY-MM-DD, YYYY-MM, YYYY, Month YYYY, Q1 2024, etc.
    Language-agnostic for date formats, English month names only.
    """
    if not text:
        return 0
    matches = _TIME_REGEX.findall(text)
    return len(matches)


def extract_time_mentions(text: str) -> list[str]:
    """
    Extract time mention strings from text.
    
    Returns list of matched strings for debugging.
    """
    if not text:
        return []
    return [m.group() for m in _TIME_REGEX.finditer(text)]


# Patterns for number mentions
# Match significant numbers (2+ digits) with optional decimal, currency, percentage
NUMBER_PATTERNS = [
    # Currency amounts: $5.2 billion, €100 million
    r"[$€£¥]\s*\d+(?:[.,]\d+)?(?:\s*(?:billion|million|thousand|B|M|K))?",
    # Percentages: 5.2%, 100%
    r"\d+(?:[.,]\d+)?\s*%",
    # Numbers with units: 100 million, 5.2 billion
    r"\d+(?:[.,]\d+)?\s*(?:billion|million|thousand|B|M|K)\b",
    # Plain significant numbers (4+ digits or decimal with 2+ digits)
    r"\b\d{4,}(?:[.,]\d+)?\b",
    r"\b\d+[.,]\d{2,}\b",
]

_NUMBER_REGEX = re.compile("|".join(f"({p})" for p in NUMBER_PATTERNS), re.IGNORECASE)


def extract_number_mentions_count(text: str) -> int:
    """
    Count significant number mentions in text.
    
    Detects: currency amounts, percentages, large numbers, decimals.
    Filters out years (handled by time patterns) and trivial numbers.
    """
    if not text:
        return 0
    
    matches = _NUMBER_REGEX.findall(text)
    # Filter out matches that look like years (4 digits between 1900-2100)
    count = 0
    for match in matches:
        # match is a tuple of groups, find non-empty one
        matched_str = next((g for g in match if g), "")
        if matched_str:
            # Skip if it's likely a year
            if re.match(r"^(19|20)\d{2}$", matched_str.strip()):
                continue
            count += 1
    return count


def extract_number_mentions(text: str) -> list[str]:
    """
    Extract number mention strings from text.
    
    Returns list of matched strings for debugging.
    """
    if not text:
        return []
    result = []
    for match in _NUMBER_REGEX.finditer(text):
        matched_str = match.group()
        # Skip years
        if re.match(r"^(19|20)\d{2}$", matched_str.strip()):
            continue
        result.append(matched_str)
    return result


# Patterns for quote marks
QUOTE_PATTERNS = [
    r'"[^"]{5,}"',      # "quote" (at least 5 chars)
    r'"[^"]{5,}"',      # "curly quotes"
    r"'[^']{5,}'",      # 'single curly quotes'
    r"«[^»]{5,}»",      # «guillemets»
    r'„[^"]{5,}"',      # „German quotes"
]

_QUOTE_REGEX = re.compile("|".join(QUOTE_PATTERNS))


def detect_quote_spans_count(text: str) -> int:
    """
    Count quote spans (quoted text) in input.
    
    Detects various quotation mark styles.
    Minimum 5 chars inside quotes to filter noise.
    """
    if not text:
        return 0
    return len(_QUOTE_REGEX.findall(text))


def detect_quote_spans(text: str) -> list[str]:
    """
    Extract quoted text spans.
    
    Returns list of matched strings for debugging.
    """
    if not text:
        return []
    return _QUOTE_REGEX.findall(text)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage Validation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CoverageAnalysis:
    """Results of coverage analysis on input text."""
    
    detected_times: int
    detected_numbers: int
    detected_quotes: int
    
    def to_dict(self) -> dict[str, int]:
        return {
            "detected_times": self.detected_times,
            "detected_numbers": self.detected_numbers,
            "detected_quotes": self.detected_quotes,
        }


def analyze_text_coverage(text: str) -> CoverageAnalysis:
    """
    Analyze text for coverage indicators.
    
    Returns counts of time mentions, number mentions, quote spans.
    """
    return CoverageAnalysis(
        detected_times=extract_time_mentions_count(text),
        detected_numbers=extract_number_mentions_count(text),
        detected_quotes=detect_quote_spans_count(text),
    )


def validate_skeleton_coverage(
    skeleton: CoverageSkeleton,
    analysis: CoverageAnalysis,
    tolerance: float = 0.5,
) -> tuple[bool, list[str]]:
    """
    Validate that skeleton covers detected content.
    
    Args:
        skeleton: Extracted skeleton
        analysis: Coverage analysis of input text
        tolerance: Fraction of detected items that can be missing (0.5 = 50%)
    
    Returns:
        (ok, reason_codes): ok=True if coverage is acceptable
    """
    reason_codes: list[str] = []
    
    # Count skeleton items with time anchors
    skeleton_times = sum(
        1 for e in skeleton.events if e.time_anchor
    ) + sum(
        1 for m in skeleton.measurements if m.time_anchor
    ) + sum(
        1 for p in skeleton.policies if p.time_anchor
    )
    
    # Count skeleton items with quantities
    skeleton_numbers = sum(
        len(m.quantity_mentions) for m in skeleton.measurements
    )
    
    # Count quotes
    skeleton_quotes = len(skeleton.quotes)
    
    # Check coverage
    if analysis.detected_times > 0:
        coverage = skeleton_times / analysis.detected_times
        if coverage < tolerance:
            reason_codes.append(
                f"low_time_coverage:{skeleton_times}/{analysis.detected_times}"
            )
    
    if analysis.detected_numbers > 0:
        coverage = skeleton_numbers / analysis.detected_numbers
        if coverage < tolerance:
            reason_codes.append(
                f"low_number_coverage:{skeleton_numbers}/{analysis.detected_numbers}"
            )
    
    if analysis.detected_quotes > 0:
        coverage = skeleton_quotes / analysis.detected_quotes
        if coverage < tolerance:
            reason_codes.append(
                f"low_quote_coverage:{skeleton_quotes}/{analysis.detected_quotes}"
            )
    
    ok = len(reason_codes) == 0
    return ok, reason_codes


# ─────────────────────────────────────────────────────────────────────────────
# Trace Events
# ─────────────────────────────────────────────────────────────────────────────


def trace_skeleton_created(skeleton: CoverageSkeleton) -> None:
    """Log skeleton creation trace event."""
    Trace.event(
        "claims.skeleton.created",
        skeleton.counts(),
    )


def trace_coverage_warning(
    analysis: CoverageAnalysis,
    skeleton: CoverageSkeleton,
    reason_codes: list[str],
) -> None:
    """Log coverage warning trace event."""
    Trace.event(
        "claims.coverage.warning",
        {
            "detected": analysis.to_dict(),
            "skeleton_counts": skeleton.counts(),
            "reason_codes": reason_codes,
        },
    )


def trace_skeleton_to_claims(
    skeleton_count: int,
    claims_emitted: int,
    claims_dropped: int,
) -> None:
    """Log skeleton to claims conversion trace event."""
    Trace.event(
        "claims.skeleton.to_claims",
        {
            "skeleton_count": skeleton_count,
            "claims_emitted": claims_emitted,
            "claims_dropped": claims_dropped,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton Parsing (from LLM response)
# ─────────────────────────────────────────────────────────────────────────────


def parse_skeleton_response(data: dict[str, Any]) -> CoverageSkeleton:
    """
    Parse LLM response into CoverageSkeleton.
    
    Handles missing/malformed fields gracefully.
    """
    events: list[SkeletonEvent] = []
    measurements: list[SkeletonMeasurement] = []
    quotes: list[SkeletonQuote] = []
    policies: list[SkeletonPolicy] = []
    
    # Parse events
    for i, e in enumerate(data.get("events", [])):
        if not isinstance(e, dict):
            continue
        events.append(SkeletonEvent(
            id=e.get("id", f"evt_{i+1}"),
            subject_entities=e.get("subject_entities", []),
            verb_phrase=e.get("verb_phrase", ""),
            time_anchor=e.get("time_anchor"),
            location_anchor=e.get("location_anchor"),
            raw_span=e.get("raw_span", ""),
        ))
    
    # Parse measurements
    for i, m in enumerate(data.get("measurements", [])):
        if not isinstance(m, dict):
            continue
        quantities = []
        for q in m.get("quantity_mentions", []):
            if isinstance(q, dict):
                quantities.append(QuantityMention(
                    value=str(q.get("value", "")),
                    unit=q.get("unit"),
                ))
        measurements.append(SkeletonMeasurement(
            id=m.get("id", f"msr_{i+1}"),
            subject_entities=m.get("subject_entities", []),
            metric=m.get("metric", ""),
            quantity_mentions=quantities,
            time_anchor=m.get("time_anchor"),
            raw_span=m.get("raw_span", ""),
        ))
    
    # Parse quotes
    for i, q in enumerate(data.get("quotes", [])):
        if not isinstance(q, dict):
            continue
        quotes.append(SkeletonQuote(
            id=q.get("id", f"qot_{i+1}"),
            speaker_entities=q.get("speaker_entities", []),
            quote_text=q.get("quote_text", ""),
            raw_span=q.get("raw_span", ""),
        ))
    
    # Parse policies
    for i, p in enumerate(data.get("policies", [])):
        if not isinstance(p, dict):
            continue
        policies.append(SkeletonPolicy(
            id=p.get("id", f"pol_{i+1}"),
            subject_entities=p.get("subject_entities", []),
            policy_action=p.get("policy_action", ""),
            time_anchor=p.get("time_anchor"),
            raw_span=p.get("raw_span", ""),
        ))
    
    return CoverageSkeleton(
        events=events,
        measurements=measurements,
        quotes=quotes,
        policies=policies,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton → Claims Conversion
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SkeletonClaimResult:
    """Result of converting a skeleton item to a claim-compatible dict."""
    
    claim_id: str
    """Generated claim ID (e.g., 'c1')."""
    
    skeleton_item_id: str
    """Original skeleton item ID (e.g., 'evt_1', 'msr_1')."""
    
    skeleton_type: str
    """Type of skeleton item: 'event', 'measurement', 'quote', 'policy'."""
    
    claim_data: dict[str, Any]
    """Claim-compatible dict for downstream enrichment."""
    
    is_valid: bool
    """Whether the claim passes basic validation."""
    
    drop_reason_codes: list[str]
    """Reason codes if claim is dropped."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "skeleton_item_id": self.skeleton_item_id,
            "skeleton_type": self.skeleton_type,
            "is_valid": self.is_valid,
            "drop_reason_codes": self.drop_reason_codes,
        }


def _skeleton_event_to_claim(
    event: SkeletonEvent,
    claim_id: str,
) -> SkeletonClaimResult:
    """Convert a skeleton event to a claim-compatible dict."""
    reason_codes: list[str] = []
    
    # Validate required fields
    if not event.subject_entities:
        reason_codes.append("missing_subject_entities")
    if not event.raw_span or len(event.raw_span) < 10:
        reason_codes.append("raw_span_too_short")
    if not event.verb_phrase:
        reason_codes.append("missing_verb_phrase")
    
    # Build retrieval seed terms from entities + verb phrase
    seed_terms = list(event.subject_entities)
    if event.verb_phrase:
        # Extract key words from verb phrase
        verb_words = [w for w in event.verb_phrase.split() if len(w) >= 3]
        seed_terms.extend(verb_words[:3])
    if event.location_anchor and event.location_anchor != "unknown":
        seed_terms.append(event.location_anchor)
    
    # Ensure minimum seed terms
    if len(seed_terms) < 3:
        reason_codes.append("insufficient_retrieval_seed_terms")
    
    # Build normalized text from raw_span
    normalized = event.raw_span
    
    claim_data = {
        "claim_text": event.raw_span,
        "text": event.raw_span,  # Compatibility
        "normalized_text": normalized,
        "subject_entities": event.subject_entities,
        "predicate_type": "event",
        "time_anchor": event.time_anchor or {"type": "unknown", "value": "unknown"},
        "location_anchor": event.location_anchor or "unknown",
        "falsifiability": {
            "is_falsifiable": True,
            "falsifiable_by": "reputable_news",
        },
        "retrieval_seed_terms": seed_terms[:10],
        "skeleton_item_id": event.id,
        "importance": 0.7,
    }
    
    return SkeletonClaimResult(
        claim_id=claim_id,
        skeleton_item_id=event.id,
        skeleton_type="event",
        claim_data=claim_data,
        is_valid=len(reason_codes) == 0,
        drop_reason_codes=reason_codes,
    )


def _skeleton_measurement_to_claim(
    measurement: SkeletonMeasurement,
    claim_id: str,
) -> SkeletonClaimResult:
    """Convert a skeleton measurement to a claim-compatible dict."""
    reason_codes: list[str] = []
    
    # Validate required fields
    if not measurement.subject_entities:
        reason_codes.append("missing_subject_entities")
    if not measurement.raw_span or len(measurement.raw_span) < 10:
        reason_codes.append("raw_span_too_short")
    if not measurement.quantity_mentions:
        reason_codes.append("missing_quantity")
    
    # Build retrieval seed terms
    seed_terms = list(measurement.subject_entities)
    if measurement.metric:
        seed_terms.append(measurement.metric)
    for q in measurement.quantity_mentions:
        if q.value:
            seed_terms.append(q.value)
        if q.unit:
            seed_terms.append(q.unit)
    
    if len(seed_terms) < 3:
        reason_codes.append("insufficient_retrieval_seed_terms")
    
    claim_data = {
        "claim_text": measurement.raw_span,
        "text": measurement.raw_span,
        "normalized_text": measurement.raw_span,
        "subject_entities": measurement.subject_entities,
        "predicate_type": "measurement",
        "time_anchor": measurement.time_anchor or {"type": "unknown", "value": "unknown"},
        "location_anchor": "unknown",
        "falsifiability": {
            "is_falsifiable": True,
            "falsifiable_by": "dataset",
        },
        "retrieval_seed_terms": seed_terms[:10],
        "skeleton_item_id": measurement.id,
        "importance": 0.8,
    }
    
    return SkeletonClaimResult(
        claim_id=claim_id,
        skeleton_item_id=measurement.id,
        skeleton_type="measurement",
        claim_data=claim_data,
        is_valid=len(reason_codes) == 0,
        drop_reason_codes=reason_codes,
    )


def _skeleton_quote_to_claim(
    quote: SkeletonQuote,
    claim_id: str,
) -> SkeletonClaimResult:
    """Convert a skeleton quote to a claim-compatible dict."""
    reason_codes: list[str] = []
    
    # Validate required fields
    if not quote.speaker_entities:
        reason_codes.append("missing_speaker_entities")
    if not quote.raw_span or len(quote.raw_span) < 10:
        reason_codes.append("raw_span_too_short")
    if not quote.quote_text:
        reason_codes.append("missing_quote_text")
    
    # Build retrieval seed terms
    seed_terms = list(quote.speaker_entities)
    # Extract key words from quote text
    quote_words = [w for w in quote.quote_text.split() if len(w) >= 4]
    seed_terms.extend(quote_words[:5])
    
    if len(seed_terms) < 3:
        reason_codes.append("insufficient_retrieval_seed_terms")
    
    claim_data = {
        "claim_text": quote.raw_span,
        "text": quote.raw_span,
        "normalized_text": quote.raw_span,
        "subject_entities": quote.speaker_entities,
        "predicate_type": "quote",
        # Quote claims don't require time_anchor (exempt)
        "time_anchor": {"type": "unknown", "value": "unknown"},
        "location_anchor": "unknown",
        "falsifiability": {
            "is_falsifiable": True,
            "falsifiable_by": "official_statement",
        },
        "retrieval_seed_terms": seed_terms[:10],
        "skeleton_item_id": quote.id,
        "importance": 0.6,
    }
    
    return SkeletonClaimResult(
        claim_id=claim_id,
        skeleton_item_id=quote.id,
        skeleton_type="quote",
        claim_data=claim_data,
        is_valid=len(reason_codes) == 0,
        drop_reason_codes=reason_codes,
    )


def _skeleton_policy_to_claim(
    policy: SkeletonPolicy,
    claim_id: str,
) -> SkeletonClaimResult:
    """Convert a skeleton policy to a claim-compatible dict."""
    reason_codes: list[str] = []
    
    # Validate required fields
    if not policy.subject_entities:
        reason_codes.append("missing_subject_entities")
    if not policy.raw_span or len(policy.raw_span) < 10:
        reason_codes.append("raw_span_too_short")
    if not policy.policy_action:
        reason_codes.append("missing_policy_action")
    
    # Build retrieval seed terms
    seed_terms = list(policy.subject_entities)
    # Extract key words from policy action
    action_words = [w for w in policy.policy_action.split() if len(w) >= 4]
    seed_terms.extend(action_words[:4])
    
    if len(seed_terms) < 3:
        reason_codes.append("insufficient_retrieval_seed_terms")
    
    claim_data = {
        "claim_text": policy.raw_span,
        "text": policy.raw_span,
        "normalized_text": policy.raw_span,
        "subject_entities": policy.subject_entities,
        "predicate_type": "policy",
        # Policy claims don't require time_anchor (exempt)
        "time_anchor": policy.time_anchor or {"type": "unknown", "value": "unknown"},
        "location_anchor": "unknown",
        "falsifiability": {
            "is_falsifiable": True,
            "falsifiable_by": "public_records",
        },
        "retrieval_seed_terms": seed_terms[:10],
        "skeleton_item_id": policy.id,
        "importance": 0.75,
    }
    
    return SkeletonClaimResult(
        claim_id=claim_id,
        skeleton_item_id=policy.id,
        skeleton_type="policy",
        claim_data=claim_data,
        is_valid=len(reason_codes) == 0,
        drop_reason_codes=reason_codes,
    )


def compute_document_context_pool(skeleton: CoverageSkeleton) -> list[str]:
    """
    Build document-level entity pool from all skeleton items.
    
    Used for context inheritance when claims have weak/empty subject_entities.
    Returns deduplicated list of entities, ordered by frequency (most common first).
    """
    from collections import Counter
    
    entity_counts: Counter[str] = Counter()
    
    for evt in skeleton.events:
        for entity in evt.subject_entities:
            if entity and len(entity) >= 2:
                entity_counts[entity.strip()] += 1
    
    for msr in skeleton.measurements:
        for entity in msr.subject_entities:
            if entity and len(entity) >= 2:
                entity_counts[entity.strip()] += 1
        # Include metric as entity (domain-specific)
        if msr.metric and len(msr.metric) >= 3:
            entity_counts[msr.metric.strip()] += 1
    
    for qot in skeleton.quotes:
        for entity in qot.speaker_entities:
            if entity and len(entity) >= 2:
                entity_counts[entity.strip()] += 1
    
    for pol in skeleton.policies:
        for entity in pol.subject_entities:
            if entity and len(entity) >= 2:
                entity_counts[entity.strip()] += 1
    
    # Return top 10 entities by frequency
    return [entity for entity, _ in entity_counts.most_common(10)]


def _select_context_entities(
    claim_entities: list[str],
    document_pool: list[str],
    max_context: int = 5,
) -> tuple[list[str], str]:
    """
    Select context entities for a claim based on its own entities and document pool.
    
    Args:
        claim_entities: The claim's own subject_entities
        document_pool: Document-level entity pool
        max_context: Maximum context entities to return
    
    Returns:
        (context_entities, source) where source is "document_pool", "none", or "already_rich"
    """
    # If claim has >= 2 entities, it's already contextually rich
    if len(claim_entities) >= 2:
        return [], "already_rich"
    
    # If no document pool available
    if not document_pool:
        return [], "none"
    
    # Filter out entities already in claim
    claim_set = {e.lower() for e in claim_entities}
    context = [e for e in document_pool if e.lower() not in claim_set]
    
    return context[:max_context], "document_pool"


def skeleton_to_claims(
    skeleton: CoverageSkeleton,
    start_claim_id: int = 1,
) -> tuple[list[SkeletonClaimResult], int, int]:
    """
    Convert skeleton items to claim-compatible dicts.
    
    Includes context anchoring: claims with weak entities inherit from document pool.
    
    Args:
        skeleton: The coverage skeleton to convert
        start_claim_id: Starting claim ID number (default 1)
    
    Returns:
        (results, emitted_count, dropped_count)
    """
    results: list[SkeletonClaimResult] = []
    claim_num = start_claim_id
    emitted = 0
    dropped = 0
    
    # Compute document-level context pool for context anchoring
    document_pool = compute_document_context_pool(skeleton)
    
    def _inject_context_and_trace(result: SkeletonClaimResult) -> None:
        """Inject context entities into claim and emit trace."""
        claim_entities = result.claim_data.get("subject_entities", [])
        context_entities, source = _select_context_entities(claim_entities, document_pool)
        
        # Always set context_entities field (may be empty)
        result.claim_data["context_entities"] = context_entities
        
        # Emit trace event for context anchoring
        if context_entities:
            Trace.event("claims.context.anchored", {
                "claim_id": result.claim_id,
                "added_context_entities": context_entities,
                "source": source,
            })
    
    # Convert events
    for event in skeleton.events:
        result = _skeleton_event_to_claim(event, f"c{claim_num}")
        _inject_context_and_trace(result)
        results.append(result)
        claim_num += 1
        if result.is_valid:
            emitted += 1
        else:
            dropped += 1
            Trace.event("claim.dropped", {
                "skeleton_item_id": result.skeleton_item_id,
                "skeleton_type": result.skeleton_type,
                "reason_codes": result.drop_reason_codes,
            })
    
    # Convert measurements
    for measurement in skeleton.measurements:
        result = _skeleton_measurement_to_claim(measurement, f"c{claim_num}")
        _inject_context_and_trace(result)
        results.append(result)
        claim_num += 1
        if result.is_valid:
            emitted += 1
        else:
            dropped += 1
            Trace.event("claim.dropped", {
                "skeleton_item_id": result.skeleton_item_id,
                "skeleton_type": result.skeleton_type,
                "reason_codes": result.drop_reason_codes,
            })
    
    # Convert quotes
    for quote in skeleton.quotes:
        result = _skeleton_quote_to_claim(quote, f"c{claim_num}")
        _inject_context_and_trace(result)
        results.append(result)
        claim_num += 1
        if result.is_valid:
            emitted += 1
        else:
            dropped += 1
            Trace.event("claim.dropped", {
                "skeleton_item_id": result.skeleton_item_id,
                "skeleton_type": result.skeleton_type,
                "reason_codes": result.drop_reason_codes,
            })
    
    # Convert policies
    for policy in skeleton.policies:
        result = _skeleton_policy_to_claim(policy, f"c{claim_num}")
        _inject_context_and_trace(result)
        results.append(result)
        claim_num += 1
        if result.is_valid:
            emitted += 1
        else:
            dropped += 1
            Trace.event("claim.dropped", {
                "skeleton_item_id": result.skeleton_item_id,
                "skeleton_type": result.skeleton_type,
                "reason_codes": result.drop_reason_codes,
            })
    
    # Emit summary trace
    trace_skeleton_to_claims(
        skeleton_count=skeleton.total_items(),
        claims_emitted=emitted,
        claims_dropped=dropped,
    )
    
    return results, emitted, dropped