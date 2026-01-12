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

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
from typing import Any, Awaitable, Callable, Dict, Iterable, Literal
from urllib.parse import urlparse, urlunparse

from spectrue_core.utils.trace import Trace

NormalizedUrl = str
ClaimId = str


@dataclass
class UrlMeta:
    status: Literal["seen", "extracted", "failed"]
    first_seen_stage: int
    source_id: str
    seen_by_claims: set[ClaimId] = field(default_factory=set)


@dataclass
class GlobalUrlRegistry:
    urls: dict[NormalizedUrl, UrlMeta] = field(default_factory=dict)


@dataclass
class ExtractedContent:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractorQueue:
    pending: list[NormalizedUrl] = field(default_factory=list)
    extracted: dict[NormalizedUrl, ExtractedContent] = field(default_factory=dict)


@dataclass
class ClaimBindings:
    eligible: dict[ClaimId, set[NormalizedUrl]] = field(default_factory=dict)
    audited: dict[ClaimId, set[NormalizedUrl]] = field(default_factory=dict)


@dataclass
class FixedPipelineState:
    registry: GlobalUrlRegistry
    extractor_queue: ExtractorQueue
    bindings: ClaimBindings


ExtractBatch = Callable[[list[NormalizedUrl]], Awaitable[dict[NormalizedUrl, ExtractedContent]]]
CheapMatch = Callable[[ClaimId, NormalizedUrl, dict[str, Any]], bool]
AuditMatch = Callable[[ClaimId, ExtractedContent], bool]

_state_ctx: ContextVar[FixedPipelineState | None] = ContextVar(
    "fixed_retrieval_state",
    default=None,
)
_extract_batch_ctx: ContextVar[ExtractBatch | None] = ContextVar(
    "fixed_retrieval_extract_batch",
    default=None,
)
_stage_ctx: ContextVar[int | None] = ContextVar("fixed_retrieval_stage", default=None)
_cheap_match_ctx: ContextVar[CheapMatch | None] = ContextVar(
    "fixed_retrieval_cheap_match",
    default=None,
)
_audit_match_ctx: ContextVar[AuditMatch | None] = ContextVar(
    "fixed_retrieval_audit_match",
    default=None,
)


class FixedPipelineContext:
    """Context manager for fixed retrieval pipeline state and callbacks."""

    def __init__(
        self,
        *,
        state: FixedPipelineState,
        extract_batch: ExtractBatch,
        cheap_match: CheapMatch | None = None,
        audit_match: AuditMatch | None = None,
        stage: int | None = None,
    ) -> None:
        self._state_token = _state_ctx.set(state)
        self._extract_token = _extract_batch_ctx.set(extract_batch)
        self._stage_token = _stage_ctx.set(stage)
        self._cheap_token = _cheap_match_ctx.set(cheap_match)
        self._audit_token = _audit_match_ctx.set(audit_match)

    def set_stage(self, stage: int) -> None:
        _stage_ctx.set(stage)

    def __enter__(self) -> "FixedPipelineContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _state_ctx.reset(self._state_token)
        _extract_batch_ctx.reset(self._extract_token)
        _stage_ctx.reset(self._stage_token)
        _cheap_match_ctx.reset(self._cheap_token)
        _audit_match_ctx.reset(self._audit_token)


def init_state() -> FixedPipelineState:
    return FixedPipelineState(
        registry=GlobalUrlRegistry(),
        extractor_queue=ExtractorQueue(),
        bindings=ClaimBindings(),
    )


def current_stage() -> int | None:
    return _stage_ctx.get()


def normalize_url(url: str) -> NormalizedUrl:
    raw = (url or "").strip()
    if not raw:
        return ""
    raw = raw.split("#", 1)[0]
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw
    if not parsed.netloc:
        return raw.rstrip("/") if raw != "/" else raw
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    if ":" in netloc:
        host, port = netloc.rsplit(":", 1)
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            netloc = host
    path = parsed.path or ""
    if path.endswith("/") and path != "/":
        path = path[:-1]
    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))


def source_id_for_url(url: str) -> str:
    nurl = normalize_url(url)
    if not nurl:
        return ""
    return hashlib.sha256(nurl.encode()).hexdigest()[:12]


def register_urls(
    stage: int,
    claim_ids: set[ClaimId],
    urls: list[str],
) -> None:
    state = _state_ctx.get()
    if state is None:
        raise RuntimeError("Fixed pipeline state is not initialized")

    registry = state.registry
    extractor_queue = state.extractor_queue
    registered = 0

    for url in urls:
        nurl = normalize_url(url)
        if not nurl:
            continue
        if nurl not in registry.urls:
            registry.urls[nurl] = UrlMeta(
                status="seen",
                first_seen_stage=stage,
                source_id=source_id_for_url(nurl),
                seen_by_claims=set(claim_ids),
            )
            extractor_queue.pending.append(nurl)
            registered += 1
        else:
            registry.urls[nurl].seen_by_claims |= set(claim_ids)

    Trace.event("urls_registered", {"stage": stage, "count": registered})


def _chunk_urls(urls: list[NormalizedUrl], batch_size: int) -> list[list[NormalizedUrl]]:
    return [urls[i : i + batch_size] for i in range(0, len(urls), batch_size)]


def _default_match(_: ClaimId, __: NormalizedUrl, ___: dict[str, Any]) -> bool:
    return True


def _default_audit(_: ClaimId, __: ExtractedContent) -> bool:
    return True


async def extract_all_batches(batch_size: int = 5) -> None:
    state = _state_ctx.get()
    if state is None:
        raise RuntimeError("Fixed pipeline state is not initialized")
    extract_batch = _extract_batch_ctx.get()
    if extract_batch is None:
        raise RuntimeError("Fixed pipeline extract_batch is not configured")

    if batch_size != 5:
        batch_size = 5

    extractor_queue = state.extractor_queue
    registry = state.registry

    urls = extractor_queue.pending
    pending_before = len(urls)
    extractor_queue.pending = []
    processed = 0
    stage = _stage_ctx.get()

    for batch in _chunk_urls(urls, batch_size):
        batch_count = len(batch)
        pending_after = max(pending_before - processed - batch_count, 0)
        Trace.event(
            "extract_batch_started",
            {
                "batch_size": batch_size,
                "pending_before": pending_before,
                "batch_urls_count": batch_count,
                "pending_after": pending_after,
                "stage": stage,
            },
        )
        result = await extract_batch(batch)
        success_count = 0
        for url, content in result.items():
            extractor_queue.extracted[url] = content
            if url in registry.urls:
                registry.urls[url].status = "extracted"
            success_count += 1
        Trace.event(
            "extract_batch_finished",
            {
                "success_count": success_count,
                "pending_before": pending_before,
                "batch_urls_count": batch_count,
                "pending_after": pending_after,
                "stage": stage,
            },
        )
        processed += batch_count


def bind_after_extract() -> None:
    state = _state_ctx.get()
    if state is None:
        raise RuntimeError("Fixed pipeline state is not initialized")

    registry = state.registry
    extractor_queue = state.extractor_queue
    bindings = state.bindings

    cheap_match = _cheap_match_ctx.get() or _default_match
    audit_match = _audit_match_ctx.get() or _default_audit

    for url, content in extractor_queue.extracted.items():
        candidate_claims = registry.urls[url].seen_by_claims
        for claim_id in candidate_claims:
            if cheap_match(claim_id, url, content.metadata):
                bindings.eligible.setdefault(claim_id, set()).add(url)

    for claim_id, urls in bindings.eligible.items():
        for url in urls:
            content = extractor_queue.extracted.get(url)
            if content and audit_match(claim_id, content):
                bindings.audited.setdefault(claim_id, set()).add(url)

    stage = _stage_ctx.get()
    Trace.event("bind_completed", {"stage": stage})


def _meta_value(source: Any, key: str, default: float) -> float:
    if isinstance(source, dict) and key in source:
        try:
            return float(source.get(key, default))
        except Exception:
            return default
    if hasattr(source, key):
        try:
            return float(getattr(source, key))
        except Exception:
            return default
    return default


def compute_sufficiency(claim_metadata: Any) -> float:
    w1 = _meta_value(claim_metadata, "w1", 1.0)
    w2 = _meta_value(claim_metadata, "w2", 1.0)
    w3 = _meta_value(claim_metadata, "w3", 1.0)
    w4 = _meta_value(claim_metadata, "w4", 1.0)
    ce_cluster_count = _meta_value(claim_metadata, "CE_cluster_count", 0.0)
    se_support_mass = _meta_value(claim_metadata, "SE_support_mass", 0.0)
    conflict_mass = _meta_value(claim_metadata, "conflict_mass", 0.0)
    missing_constraints = _meta_value(claim_metadata, "missing_constraints", 0.0)
    return (
        w1 * ce_cluster_count
        + w2 * se_support_mass
        - w3 * conflict_mass
        - w4 * missing_constraints
    )
