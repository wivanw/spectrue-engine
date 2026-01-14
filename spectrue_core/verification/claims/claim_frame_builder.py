# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
ClaimFrame builder for per-claim deep analysis.

Assembles ClaimFrame objects from claim data, document structure,
evidence items, and execution state.
"""

from __future__ import annotations

import hashlib
from typing import Any

from spectrue_core.schema.claim_frame import (
    ClaimFrame,
    ConfirmationCounts,
    ContextExcerpt,
    ContextMeta,
    EvidenceItemFrame,
)
from spectrue_core.utils.text_structure import (
    TextStructure,
    extract_text_structure,
)
from spectrue_core.verification.evidence.evidence_stats import (
    build_confirmation_counts,
    build_evidence_stats,
)
from spectrue_core.verification.search.retrieval_trace import (
    create_empty_retrieval_trace,
    format_retrieval_trace,
)
from spectrue_core.verification.orchestration.execution_plan import ClaimExecutionState
from spectrue_core.verification.retrieval.fixed_pipeline import source_id_for_url


def _generate_evidence_id(claim_id: str, url: str, index: int) -> str:
    """Generate a unique evidence ID from claim, URL, and index."""
    hash_input = f"{claim_id}:{url}:{index}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


def build_context_excerpt(
    document_text: str,
    claim_text: str,
    structure: TextStructure | None = None,
    window_size: int = 1,
) -> tuple[ContextExcerpt, ContextMeta]:
    """
    Build context excerpt and metadata for a claim.
    
    Uses layered approach:
    1. Sentence-window (claim sentence + neighbors)
    2. Paragraph-level fallback
    3. Character-based fallback
    
    Args:
        document_text: Full document text
        claim_text: The claim text to find context for
        structure: Pre-computed text structure (optional)
        window_size: Number of sentences on each side of claim (default 1)
    
    Returns:
        Tuple of (ContextExcerpt, ContextMeta)
    """
    from spectrue_core.utils.context_excerpt import (
        build_context_excerpt as _build_excerpt,
    )

    # Use the improved context excerpt builder
    result = _build_excerpt(
        document_text=document_text,
        claim_text=claim_text,
        window_size=window_size,
        max_chars=500,
    )

    # Generate document ID from content hash
    doc_id = hashlib.sha256(document_text[:1000].encode()).hexdigest()[:16]

    context_excerpt = ContextExcerpt(
        text=result.text,
        source_type="user_text",
        span_start=result.span_start,
        span_end=result.span_end,
    )

    context_meta = ContextMeta(
        document_id=doc_id,
        paragraph_index=result.paragraph_index,
        sentence_index=result.sentence_index,
        sentence_window=result.sentence_window,
    )

    return context_excerpt, context_meta



def convert_evidence_items(
    claim_id: str,
    raw_evidence: list[dict[str, Any]],
) -> tuple[EvidenceItemFrame, ...]:
    """
    Convert raw evidence dicts to EvidenceItemFrame objects.
    
    Args:
        claim_id: The claim these evidence items belong to
        raw_evidence: List of evidence dicts from pipeline
    
    Returns:
        Tuple of EvidenceItemFrame objects
    """
    items: list[EvidenceItemFrame] = []

    for idx, ev in enumerate(raw_evidence):
        evidence_id = _generate_evidence_id(
            claim_id, 
            ev.get("url", f"unknown_{idx}"),
            idx
        )
        source_id = str(ev.get("source_id") or "")
        if not source_id:
            source_id = source_id_for_url(ev.get("url", "")) or evidence_id

        item = EvidenceItemFrame(
            evidence_id=evidence_id,
            claim_id=claim_id,
            url=ev.get("url", ""),
            source_id=source_id,
            title=ev.get("title"),
            source_tier=ev.get("tier") or ev.get("source_tier"),
            source_type=ev.get("source_type"),
            stance=ev.get("stance"),
            quote=ev.get("quote"),
            snippet=ev.get("snippet") or ev.get("content"),
            relevance=ev.get("relevance") or ev.get("score"),
            content_hash=ev.get("content_hash"),
            publisher_id=ev.get("publisher_id"),
            similar_cluster_id=ev.get("similar_cluster_id"),
            attribution=ev.get("attribution"),
        )
        items.append(item)

    return tuple(items)


def build_claim_frame(
    claim_id: str,
    claim_text: str,
    claim_language: str,
    document_text: str,
    raw_evidence: list[dict[str, Any]],
    execution_state: ClaimExecutionState | None = None,
    structure: TextStructure | None = None,
    window_size: int = 1,
    confirmation_lambda: float | None = None,
) -> ClaimFrame:
    """
    Build a complete ClaimFrame for deep analysis.
    
    Args:
        claim_id: Unique claim identifier
        claim_text: The claim text
        claim_language: ISO-639-1 language code
        document_text: Full source document text
        raw_evidence: List of evidence dicts scoped to this claim
        execution_state: Retrieval execution state (optional)
        structure: Pre-computed text structure (optional)
        window_size: Sentence window size for context (default 1)
    
    Returns:
        Complete ClaimFrame ready for judge invocation
    """
    # Build context
    context_excerpt, context_meta = build_context_excerpt(
        document_text=document_text,
        claim_text=claim_text,
        structure=structure,
        window_size=window_size,
    )

    # Convert evidence items
    evidence_items = convert_evidence_items(claim_id, raw_evidence)

    # Build evidence stats + confirmation counts
    evidence_stats = build_evidence_stats(evidence_items)
    if confirmation_lambda is not None:
        confirmation_counts = build_confirmation_counts(
            evidence_items,
            lambda_weight=confirmation_lambda,
        )
    else:
        confirmation_counts = ConfirmationCounts()

    # Build retrieval trace
    if execution_state is not None:
        retrieval_trace = format_retrieval_trace(execution_state)
    else:
        retrieval_trace = create_empty_retrieval_trace()

    return ClaimFrame(
        claim_id=claim_id,
        claim_text=claim_text,
        claim_language=claim_language.lower()[:2],  # Normalize to 2-char
        context_excerpt=context_excerpt,
        context_meta=context_meta,
        evidence_items=evidence_items,
        evidence_stats=evidence_stats,
        confirmation_counts=confirmation_counts,
        retrieval_trace=retrieval_trace,
    )


def build_claim_frames_from_pipeline(
    claims: list[dict[str, Any]],
    document_text: str,
    evidence_by_claim: dict[str, list[dict[str, Any]]],
    execution_states: dict[str, ClaimExecutionState] | None = None,
    confirmation_lambda: float | None = None,
) -> list[ClaimFrame]:
    """
    Build ClaimFrame objects for all claims from pipeline data.
    
    Args:
        claims: List of claim dicts from extraction
        document_text: Full source document text
        evidence_by_claim: Dict mapping claim_id to evidence list
        execution_states: Dict mapping claim_id to execution state
    
    Returns:
        List of ClaimFrame objects
    """
    # Pre-compute structure once
    structure = extract_text_structure(document_text)

    frames: list[ClaimFrame] = []

    for claim in claims:
        claim_id = claim.get("claim_id") or claim.get("id") or str(len(frames))
        claim_text = claim.get("text") or claim.get("normalized_text") or ""
        claim_lang = claim.get("language") or claim.get("claim_language") or "en"

        raw_evidence = evidence_by_claim.get(claim_id, [])
        exec_state = execution_states.get(claim_id) if execution_states else None

        frame = build_claim_frame(
            claim_id=claim_id,
            claim_text=claim_text,
            claim_language=claim_lang,
            document_text=document_text,
            raw_evidence=raw_evidence,
            execution_state=exec_state,
            structure=structure,
            confirmation_lambda=confirmation_lambda,
        )
        frames.append(frame)

    return frames
