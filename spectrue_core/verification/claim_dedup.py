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

from dataclasses import dataclass

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


def dedup_claims_post_extraction(
    claims: list[dict],
    *,
    tau: float = 0.90,
) -> tuple[list[dict], list[DedupInfo]]:
    """
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

    canonical_claims: list[dict] = []
    canonical_texts: list[str] = []
    dedup_pairs: list[DedupInfo] = []

    # We keep aliases on the canonical claim for UI/trace (non-breaking extra fields).
    # Structure:
    #   canonical["aliases"] = [{"id": "...", "text": "...", "sim": 0.93}, ...]
    #   canonical["dedup_group_size"] = 1 + len(aliases)
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
