# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

"""
Claim Dedup (Embeddings) — post-extraction, pre-orchestration.

Goal:
  Normalize the claim set BEFORE oracle/graph/search/scoring so we don't waste budget on
  duplicates and we don't confuse anchor selection.

Contract:
  - No text heuristics (no substring rules).
  - Deterministic greedy clustering using embeddings + cosine similarity.
  - Canonical claim keeps its original fields; duplicates are attached as `aliases` metadata.
  - If embeddings are unavailable => no-op.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
from functools import partial

from spectrue_core.utils.embedding_service import EmbedService


def _claim_id(c: dict) -> str:
    return str(c.get("id") or c.get("claim_id") or "").strip()


def _claim_text(c: dict) -> str:
    # Extraction may use different keys across versions; keep it robust.
    t = c.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    t = c.get("claim_text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    t = c.get("statement")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return ""


@dataclass(frozen=True)
class DedupInfo:
    canonical_id: str
    duplicate_id: str
    similarity: float


def _dedup_sync(
    claims: list[dict],
    tau: float,
) -> tuple[list[dict], list[DedupInfo]]:
    """
    Synchronous implementation of dedup logic.
    Called from thread pool to avoid blocking event loop.
    """
    canonical_claims: list[dict] = []
    canonical_texts: list[str] = []
    dedup_pairs: list[DedupInfo] = []

    for c in claims:
        if not isinstance(c, dict):
            continue

        cid = _claim_id(c)
        txt = _claim_text(c)
        # Keep malformed claims untouched (let later stages handle validation).
        if not cid or not txt:
            canonical_claims.append(c)
            canonical_texts.append(txt or "")
            continue

        # First canonical
        if not canonical_claims:
            c.setdefault("aliases", [])
            c["dedup_group_size"] = 1
            canonical_claims.append(c)
            canonical_texts.append(txt)
            continue

        # Similarity to existing canonicals
        scores = EmbedService.batch_similarity(txt, canonical_texts)
        if not scores:
            # Extremely defensive fallback (shouldn't happen)
            c.setdefault("aliases", [])
            c["dedup_group_size"] = 1
            canonical_claims.append(c)
            canonical_texts.append(txt)
            continue

        best_idx = max(range(len(scores)), key=lambda i: float(scores[i]))
        best_sim = float(scores[best_idx])

        if best_sim >= float(tau):
            canon = canonical_claims[best_idx]
            canon_id = _claim_id(canon) or f"canon_{best_idx}"
            dedup_pairs.append(
                DedupInfo(
                    canonical_id=canon_id,
                    duplicate_id=cid,
                    similarity=best_sim,
                )
            )

            aliases = canon.setdefault("aliases", [])
            if isinstance(aliases, list):
                aliases.append({"id": cid, "text": txt, "sim": best_sim})
            try:
                canon["dedup_group_size"] = 1 + len(canon.get("aliases") or [])
            except Exception:
                canon["dedup_group_size"] = 1
            continue

        # New canonical
        c.setdefault("aliases", [])
        c["dedup_group_size"] = 1
        canonical_claims.append(c)
        canonical_texts.append(txt)

    return canonical_claims, dedup_pairs


# Thread pool for embedding operations (shared across calls)
_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Get or create thread pool executor."""
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="embed_dedup",
        )
    return _executor


async def dedup_claims_post_extraction_async(
    claims: list[dict],
    *,
    tau: float = 0.90,
) -> tuple[list[dict], list[DedupInfo]]:
    """
    Async version of claim deduplication.
    
    Runs embedding operations in thread pool to avoid blocking event loop.
    
    Greedy semantic dedup:
      iterate in input order, assign each claim to the most similar canonical;
      if similarity >= tau => mark duplicate; else becomes new canonical.

    Returns:
      (canonical_claims, dedup_pairs)
    """
    if not claims:
        return [], []

    # If embeddings aren't available, don't guess with heuristics.
    if not EmbedService.is_available():
        return claims, []

    loop = asyncio.get_running_loop()
    executor = _get_executor()

    # Run sync dedup in thread pool
    result = await loop.run_in_executor(
        executor,
        partial(_dedup_sync, claims, tau),
    )

    return result


def dedup_claims_post_extraction(
    claims: list[dict],
    *,
    tau: float = 0.90,
) -> tuple[list[dict], list[DedupInfo]]:
    """
    Synchronous wrapper for backward compatibility.
    
    Greedy semantic dedup:
      iterate in input order, assign each claim to the most similar canonical;
      if similarity >= tau => mark duplicate; else becomes new canonical.

    Returns:
      (canonical_claims, dedup_pairs)
    """
    if not claims:
        return [], []

    # If embeddings aren't available, don't guess with heuristics.
    if not EmbedService.is_available():
        return claims, []

    return _dedup_sync(claims, tau)
