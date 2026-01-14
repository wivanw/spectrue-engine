# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectrue_core.graph.embedding_util import EmbeddingClient, cosine_similarity
from spectrue_core.runtime_config import DeepV2Config


@dataclass(frozen=True)
class AttributionResult:
    sources: list[dict[str, Any]]
    evidence_by_claim: dict[str, list[dict[str, Any]]]


def _claim_id_for(claim: dict[str, Any], idx: int) -> str:
    raw = claim.get("id") or claim.get("claim_id")
    if raw:
        return str(raw)
    return f"c{idx + 1}"


async def attribute_cluster_evidence(
    *,
    claims: list[dict[str, Any]],
    cluster_claims: dict[str, list[dict[str, Any]]],
    cluster_evidence_docs: dict[str, list[dict[str, Any]]],
    deep_v2_cfg: DeepV2Config,
) -> AttributionResult:
    evidence_by_claim: dict[str, list[dict[str, Any]]] = {}
    sources: list[dict[str, Any]] = []

    embedding_client = EmbeddingClient()

    for cluster_id, claim_list in cluster_claims.items():
        docs = cluster_evidence_docs.get(cluster_id, [])
        if not claim_list or not docs:
            continue

        claim_texts = [
            str(c.get("normalized_text") or c.get("text") or "")
            for c in claim_list
        ]
        claim_embeddings = await embedding_client.embed_texts(claim_texts)

        doc_embeddings = [
            d.get("embedding") for d in docs
        ]
        if any(emb is None for emb in doc_embeddings):
            doc_texts = [str(d.get("cleaned_text") or "") for d in docs]
            doc_embeddings = await embedding_client.embed_texts(doc_texts)

        for idx, claim in enumerate(claim_list):
            claim_id = _claim_id_for(claim, idx)
            evidence_by_claim.setdefault(claim_id, [])
            if idx >= len(claim_embeddings):
                continue

            scored: list[tuple[int, float]] = []
            for d_idx, doc in enumerate(docs):
                doc_emb = doc_embeddings[d_idx] if d_idx < len(doc_embeddings) else None
                if not doc_emb:
                    similarity = 0.0
                else:
                    similarity = cosine_similarity(claim_embeddings[idx], doc_emb)
                scored.append((d_idx, float(similarity)))

            scored.sort(key=lambda item: item[1], reverse=True)
            precision_k = int(deep_v2_cfg.precision_top_k)
            corr_k = int(deep_v2_cfg.corroboration_top_k)

            selected: set[int] = set()
            for d_idx, score in scored[:precision_k]:
                selected.add(d_idx)
                doc = docs[d_idx]
                evidence = _build_evidence_item(claim_id, doc, score, attribution="precision")
                evidence_by_claim[claim_id].append(evidence)
                sources.append(evidence)

            corr_added = 0
            for d_idx, score in scored:
                if d_idx in selected:
                    continue
                doc = docs[d_idx]
                evidence = _build_evidence_item(claim_id, doc, score, attribution="corroboration")
                evidence_by_claim[claim_id].append(evidence)
                sources.append(evidence)
                corr_added += 1
                if corr_added >= corr_k:
                    break

    return AttributionResult(sources=sources, evidence_by_claim=evidence_by_claim)


def _build_evidence_item(
    claim_id: str,
    doc: dict[str, Any],
    similarity: float,
    *,
    attribution: str,
) -> dict[str, Any]:
    cleaned = str(doc.get("cleaned_text") or "")
    content_excerpt = cleaned[:1500] if cleaned else ""

    return {
        "claim_id": claim_id,
        "url": doc.get("canonical_url") or doc.get("url"),
        "source_id": doc.get("source_id"),
        "title": doc.get("title"),
        "snippet": doc.get("snippet") or content_excerpt,
        "content": content_excerpt,
        "provider_score": doc.get("provider_score"),
        "similarity_score": similarity,
        "content_hash": doc.get("content_hash"),
        "publisher_id": doc.get("publisher_id"),
        "similar_cluster_id": doc.get("similar_cluster_id"),
        "attribution": attribution,
    }
