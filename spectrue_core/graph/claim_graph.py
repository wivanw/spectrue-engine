# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M72: Hybrid ClaimGraph (B + C) Builder

Two-stage sparse graph for claim prioritization:
- B-stage: cheap candidate generation (embeddings + adjacency)
- C-stage: LLM edge typing (GPT-5 nano)

Evidence-first principle: This system NEVER produces truth.
It only identifies WHICH claims to prioritize for verification.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from math import ceil
from typing import TYPE_CHECKING

from spectrue_core.graph.types import (
    CandidateEdge,
    ClaimNode,
    DedupeResult,
    EdgeRelation,
    GraphResult,
    RankedClaim,
    RELATION_MULTIPLIERS,
    STRUCTURAL_RELATIONS,
    TypedEdge,
)
from spectrue_core.graph.embedding_util import EmbeddingClient

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from spectrue_core.runtime_config import ClaimGraphConfig
    from spectrue_core.agents.skills.edge_typing import EdgeTypingSkill

logger = logging.getLogger(__name__)

# Cost estimation constants (GPT-5 nano pricing)
# ~$0.00015 per 1K input tokens, ~$0.0006 per 1K output tokens
COST_PER_INPUT_TOKEN = 0.00015 / 1000
COST_PER_OUTPUT_TOKEN = 0.0006 / 1000
EMBEDDING_COST_PER_TOKEN = 0.00002 / 1000  # text-embedding-3-small

# Latency estimation
AVG_LATENCY_PER_LLM_CALL_SEC = 3.0
AVG_LATENCY_PER_EMBEDDING_BATCH_SEC = 0.5


class ClaimGraphBuilder:
    """
    Hybrid ClaimGraph builder with B-stage (candidate) and C-stage (typing).
    
    Usage:
        builder = ClaimGraphBuilder(config, openai_client, edge_typing_skill)
        result = await builder.build(claims)
        
        if not result.disabled:
            key_claim_ids = result.key_claim_ids
    """
    
    def __init__(
        self,
        config: "ClaimGraphConfig",
        openai_client: "AsyncOpenAI",
        edge_typing_skill: "EdgeTypingSkill | None" = None,
    ):
        self.config = config
        self.embedding_client = EmbeddingClient(openai_client)
        self.edge_typing_skill = edge_typing_skill
        
        # Edge typing cache: (src_hash, dst_hash) -> TypedEdge
        self._edge_cache: dict[tuple[str, str], TypedEdge | None] = {}
    
    async def build(self, claims: list[dict]) -> GraphResult:
        """
        Build claim graph and identify key claims.
        
        Args:
            claims: List of claim dicts from extraction
            
        Returns:
            GraphResult with key claims and metrics
        """
        start_time = time.time()
        result = GraphResult(claims_count_raw=len(claims))
        
        if not claims or len(claims) < 2:
            logger.debug("[M72] ClaimGraph skipped: < 2 claims")
            # M74: Ensure fallback even if skipped
            nodes = [ClaimNode.from_claim_dict(c, i) for i, c in enumerate(claims)] if claims else []
            return self._apply_fallback(result, nodes)
        
        try:
            # Step 1: Convert to ClaimNodes
            nodes = [ClaimNode.from_claim_dict(c, i) for i, c in enumerate(claims)]
            result.claims_count_raw = len(nodes)
            
            # Step 2: Deduplicate
            dedup = self._deduplicate(nodes)
            nodes = dedup.canonical_claims
            result.claims_count_dedup = len(nodes)
            
            if len(nodes) < 2:
                logger.debug("[M72] ClaimGraph skipped after dedup: < 2 claims")
                return self._apply_fallback(result, nodes)
            
            # Step 3: B-Stage - Generate candidates
            candidates = await self._generate_candidates(nodes)
            result.candidate_edges_count = len(candidates)
            
            if not candidates:
                logger.debug("[M72] ClaimGraph: no candidate edges generated")
                # M74: Fallback
                result.disabled = True
                result.disabled_reason = "no_candidates"
                return self._apply_fallback(result, nodes)
            
            # Step 4: Pre-flight budget check
            budget_ok, estimated_cost, estimated_latency = self._preflight_budget_check(
                len(nodes), len(candidates)
            )
            
            if not budget_ok:
                result.disabled = True
                result.disabled_reason = "budget_exceeded_preflight"
                result.cost_usd = estimated_cost
                logger.warning(
                    "[M72] ClaimGraph disabled: budget exceeded (cost=$%.4f, latency=%.1fs)",
                    estimated_cost, estimated_latency
                )
                return self._apply_fallback(result, nodes)
            
            # Step 5: C-Stage - Edge typing
            typed_edges = await self._type_edges(candidates, nodes)
            
            # Step 6: Filter edges by τ and relation
            kept_edges = [
                e for e in typed_edges
                if e.relation != EdgeRelation.UNRELATED and e.score >= self.config.tau
            ]
            
            result.typed_edges_kept_count = len(kept_edges)
            result.typed_edges = kept_edges
            
            # Count by relation
            relation_counts: dict[str, int] = defaultdict(int)
            for e in typed_edges:
                relation_counts[e.relation.value] += 1
            result.typed_edges_by_relation = dict(relation_counts)
            
            # Calculate kept ratio
            # M74: Pass candidates and kept_edges to quality gate (which calculates ratio)
            
            # Step 7: Quality gate
            if not self._quality_gate(len(candidates), candidates, kept_edges, result):
                result.disabled = True
                result.disabled_reason = "quality_gate_failed"
                # Use within-topic ratio for main kept_ratio field to align with decision
                result.kept_ratio = result.kept_ratio_within_topic
                
                logger.warning(
                    "[M72] ClaimGraph disabled: quality gate failed (ratio=%.3f)",
                    result.kept_ratio
                )
                return self._apply_fallback(result, nodes)
            
            result.kept_ratio = result.kept_ratio_within_topic

            # Step 8: Graph ranking (PageRank)
            ranked = self._rank_claims(nodes, kept_edges)
            result.all_ranked = ranked
            result.key_claims = [r for r in ranked if r.is_key_claim]
            
            # Finalize metrics
            elapsed_ms = int((time.time() - start_time) * 1000)
            result.latency_ms = elapsed_ms
            result.cost_usd = self._estimate_actual_cost(len(nodes), len(candidates))
            
            logger.info(
                "[M72] ClaimGraph complete: %d claims, %d edges kept, %d key claims (%.1fs, $%.4f)",
                len(nodes), len(kept_edges), len(result.key_claims),
                elapsed_ms / 1000, result.cost_usd
            )
            
            return result
            
        except Exception as e:
            logger.warning("[M72] ClaimGraph failed: %s", e)
            result.disabled = True
            result.disabled_reason = f"error: {str(e)[:50]}"
            # Try fallback if nodes exist
            if 'nodes' in locals() and nodes:
                return self._apply_fallback(result, nodes)
            return result

    def _apply_fallback(self, result: GraphResult, nodes: list[ClaimNode]) -> GraphResult:
        """M74/T6: Apply deterministic fallback when graph is disabled."""
        if not nodes:
            return result
            
        result.fallback_used = True
        
        # Sort by importance desc, then text asc (deterministic)
        sorted_nodes = sorted(
            nodes, 
            key=lambda n: (-n.importance, n.text)
        )
        
        # Populate key_claims
        top_k = self.config.top_k
        result.key_claims = [
            RankedClaim(
                claim_id=n.claim_id,
                centrality_score=n.importance,
                in_structural_weight=0.0,
                in_contradict_weight=0.0,
                is_key_claim=True
            )
            for n in sorted_nodes[:top_k]
        ]
        
        # Populate all_ranked
        result.all_ranked = [
            RankedClaim(
                claim_id=n.claim_id,
                centrality_score=n.importance,
                in_structural_weight=0.0,
                in_contradict_weight=0.0,
                is_key_claim=(i < top_k)
            )
            for i, n in enumerate(sorted_nodes)
        ]
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Deduplication
    # ─────────────────────────────────────────────────────────────────────────
    
    def _deduplicate(self, nodes: list[ClaimNode]) -> DedupeResult:
        """
        Cluster near-duplicate claims using 90% Jaccard word overlap.
        
        Keeps canonical representative (highest importance).
        """
        if not nodes:
            return DedupeResult([], {}, 1.0)
        
        # Group by normalized text key
        groups: dict[str, list[ClaimNode]] = defaultdict(list)
        
        for node in nodes:
            # Normalize: lowercase, collapse whitespace
            key = " ".join(node.text.lower().split())
            groups[key].append(node)
        
        # For each group, find canonical (highest importance)
        canonical_claims: list[ClaimNode] = []
        dedup_map: dict[str, list[str]] = {}
        
        for key, group in groups.items():
            if len(group) == 1:
                canonical_claims.append(group[0])
                continue
            
            # Sort by importance descending
            group.sort(key=lambda n: n.importance, reverse=True)
            canonical = group[0]
            canonical_claims.append(canonical)
            
            # Map merged IDs
            merged_ids = [n.claim_id for n in group[1:]]
            if merged_ids:
                dedup_map[canonical.claim_id] = merged_ids
                logger.debug("[M72] Dedup: %s absorbed %d duplicates", 
                            canonical.claim_id, len(merged_ids))
        
        # Now do fuzzy dedup using Jaccard similarity
        canonical_claims = self._fuzzy_dedup(canonical_claims, threshold=0.9)
        
        reduction = len(nodes) / len(canonical_claims) if canonical_claims else 1.0
        
        if len(canonical_claims) < len(nodes):
            logger.info("[M72] Dedup: %d → %d claims (ratio=%.2f)", 
                       len(nodes), len(canonical_claims), reduction)
        
        return DedupeResult(canonical_claims, dedup_map, reduction)
    
    def _fuzzy_dedup(
        self, 
        nodes: list[ClaimNode], 
        threshold: float = 0.9
    ) -> list[ClaimNode]:
        """Fuzzy deduplication using Jaccard word similarity."""
        if len(nodes) <= 1:
            return nodes
        
        # Sort by importance (highest first to keep as canonical)
        sorted_nodes = sorted(nodes, key=lambda n: n.importance, reverse=True)
        kept: list[ClaimNode] = []
        
        for node in sorted_nodes:
            node_words = set(node.text.lower().split())
            if not node_words:
                continue
            
            is_duplicate = False
            for existing in kept:
                existing_words = set(existing.text.lower().split())
                if not existing_words:
                    continue
                
                # Jaccard similarity
                intersection = len(node_words & existing_words)
                union = len(node_words | existing_words)
                similarity = intersection / union if union > 0 else 0
                
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept.append(node)
        
        return kept
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: B-Stage - Candidate Generation
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _generate_candidates(
        self, 
        nodes: list[ClaimNode]
    ) -> list[CandidateEdge]:
        """
        Generate candidate edges using embedding similarity and adjacency.
        
        MVP: sim + adjacent (keyword overlap is stub/disabled)
        """
        if len(nodes) < 2:
            return []
        
        # Generate embeddings
        texts = [n.text for n in nodes]
        embeddings = await self.embedding_client.embed_texts(texts)
        
        # Build similarity matrix
        sim_matrix = self.embedding_client.build_similarity_matrix(embeddings)
        
        candidates: list[CandidateEdge] = []
        seen_pairs: set[tuple[str, str]] = set()
        
        for i, node in enumerate(nodes):
            node_candidates: list[CandidateEdge] = []
            
            # Similarity-based candidates (Top-K_SIM)
            # M74: Topic-aware check
            # We get all indices sorted by sim, then iterate and filtering by topic if needed
            top_similar = self.embedding_client.get_top_k_similar(
                i, sim_matrix, k=len(nodes)  # Get max candidates, we filter below
            )
            
            sim_count = 0
            for j, sim_score in top_similar:
                if sim_count >= self.config.k_sim:
                    break

                other = nodes[j]
                
                # M74: Skip cross-topic for SIM edges
                if self.config.topic_aware and node.topic_key != other.topic_key:
                    continue
                
                pair = tuple(sorted([node.claim_id, other.claim_id]))
                
                if pair in seen_pairs:
                    continue
                
                seen_pairs.add(pair)
                node_candidates.append(CandidateEdge(
                    src_id=node.claim_id,
                    dst_id=other.claim_id,
                    reason="sim",
                    sim_score=sim_score,
                    same_section=node.section_id == other.section_id,
                    cross_topic=False,  # SIM edges are always within-topic here
                ))
                sim_count += 1
            
            # Adjacency-based candidates (±K_ADJ in same section)
            for delta in range(-self.config.k_adj, self.config.k_adj + 1):
                if delta == 0:
                    continue
                
                adj_idx = i + delta
                if adj_idx < 0 or adj_idx >= len(nodes):
                    continue
                
                other = nodes[adj_idx]
                # Keep section constraint? Usually yes.
                if node.section_id != other.section_id:
                    continue
                
                pair = tuple(sorted([node.claim_id, other.claim_id]))
                if pair in seen_pairs:
                    continue
                
                seen_pairs.add(pair)
                
                # M74: Mark cross-topic for Adjacency
                is_cross_topic = (node.topic_key != other.topic_key)
                
                node_candidates.append(CandidateEdge(
                    src_id=node.claim_id,
                    dst_id=other.claim_id,
                    reason="adjacent",
                    sim_score=0.0,  # N/A for adjacency
                    same_section=True,
                    cross_topic=is_cross_topic,
                ))
            
            # Apply cap per node
            if len(node_candidates) > self.config.k_total_cap:
                # Prioritize by sim_score (descending)
                node_candidates.sort(key=lambda e: e.sim_score, reverse=True)
                node_candidates = node_candidates[:self.config.k_total_cap]
            
            candidates.extend(node_candidates)
        
        logger.debug("[M72] B-Stage: %d candidate edges from %d nodes", 
                    len(candidates), len(nodes))
        
        return candidates
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Pre-Flight Budget Check
    # ─────────────────────────────────────────────────────────────────────────
    
    def _preflight_budget_check(
        self,
        num_claims: int,
        num_candidates: int,
    ) -> tuple[bool, float, float]:
        """
        Estimate cost and latency before running C-stage.
        
        Returns:
            (budget_ok, estimated_cost_usd, estimated_latency_sec)
        """
        # Embedding cost
        avg_chars_per_claim = 200
        embedding_tokens = num_claims * avg_chars_per_claim // 4
        embedding_cost = embedding_tokens * EMBEDDING_COST_PER_TOKEN
        
        # Edge typing cost
        tokens_per_edge = self.config.avg_tokens_per_edge
        total_edge_tokens = num_candidates * tokens_per_edge
        # Assume 70% input, 30% output tokens
        input_tokens = int(total_edge_tokens * 0.7)
        output_tokens = int(total_edge_tokens * 0.3)
        edge_cost = (input_tokens * COST_PER_INPUT_TOKEN + 
                    output_tokens * COST_PER_OUTPUT_TOKEN)
        
        estimated_cost = embedding_cost + edge_cost
        
        # Latency estimation
        num_batches = ceil(num_candidates / self.config.batch_size)
        embedding_batches = ceil(num_claims / 20)  # ~20 claims per embedding batch
        
        estimated_latency = (
            embedding_batches * AVG_LATENCY_PER_EMBEDDING_BATCH_SEC +
            num_batches * AVG_LATENCY_PER_LLM_CALL_SEC
        )
        
        budget_ok = (
            estimated_cost <= self.config.max_cost_usd and
            estimated_latency <= self.config.max_latency_sec
        )
        
        logger.debug(
            "[M72] Pre-flight: cost=$%.4f (max=$%.2f), latency=%.1fs (max=%.0fs) → %s",
            estimated_cost, self.config.max_cost_usd,
            estimated_latency, self.config.max_latency_sec,
            "OK" if budget_ok else "EXCEEDED"
        )
        
        return budget_ok, estimated_cost, estimated_latency
    
    def _estimate_actual_cost(self, num_claims: int, num_candidates: int) -> float:
        """Estimate actual cost after execution."""
        # Same formula as preflight, but called after to track actual
        avg_chars_per_claim = 200
        embedding_tokens = num_claims * avg_chars_per_claim // 4
        embedding_cost = embedding_tokens * EMBEDDING_COST_PER_TOKEN
        
        tokens_per_edge = self.config.avg_tokens_per_edge
        total_edge_tokens = num_candidates * tokens_per_edge
        input_tokens = int(total_edge_tokens * 0.7)
        output_tokens = int(total_edge_tokens * 0.3)
        edge_cost = (input_tokens * COST_PER_INPUT_TOKEN + 
                    output_tokens * COST_PER_OUTPUT_TOKEN)
        
        return embedding_cost + edge_cost
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: C-Stage - Edge Typing
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _type_edges(
        self,
        candidates: list[CandidateEdge],
        nodes: list[ClaimNode],
    ) -> list[TypedEdge]:
        """
        Classify candidate edges using LLM (GPT-5 nano).
        
        Uses batching and caching for efficiency.
        """
        if not candidates:
            return []
        
        if not self.edge_typing_skill:
            logger.warning("[M72] No edge typing skill configured, returning empty")
            return []
        
        node_map = {n.claim_id: n for n in nodes}
        typed_edges: list[TypedEdge] = []
        edges_to_type: list[CandidateEdge] = []
        
        # Check cache
        for edge in candidates:
            cache_key = self._edge_cache_key(edge.src_id, edge.dst_id, node_map)
            if cache_key in self._edge_cache:
                cached = self._edge_cache[cache_key]
                if cached is not None:
                    typed_edges.append(cached)
            else:
                edges_to_type.append(edge)
        
        if not edges_to_type:
            logger.debug("[M72] C-Stage: all %d edges from cache", len(typed_edges))
            return typed_edges
        
        logger.debug("[M72] C-Stage: typing %d edges (%d from cache)",
                    len(edges_to_type), len(typed_edges))
        
        # Batch edges
        batches = [
            edges_to_type[i:i + self.config.batch_size]
            for i in range(0, len(edges_to_type), self.config.batch_size)
        ]
        
        for batch in batches:
            batch_results = await self.edge_typing_skill.type_edges_batch(
                batch, node_map, self.config.max_claim_chars
            )
            
            for edge, result in zip(batch, batch_results):
                cache_key = self._edge_cache_key(edge.src_id, edge.dst_id, node_map)
                self._edge_cache[cache_key] = result
                if result is not None:
                    # M75: Preserve cross_topic flag
                    result.cross_topic = edge.cross_topic
                    typed_edges.append(result)
        
        # Also fix cached edges (in case they didn't have it set, though cache stores object reference)
        # But we need to make sure we set it for this candidate's context
        # Actually TypedEdge object is shared if cached?
        # If cache persists across builds, cross_topic property is structural between nodes, so it's stable.
        # But we need to ensure it's set for cached ones too if we didn't just create them.
        for edge, result in zip(candidates, typed_edges):
             if result:
                 result.cross_topic = edge.cross_topic

        return typed_edges
    
    def _edge_cache_key(
        self, 
        src_id: str, 
        dst_id: str, 
        node_map: dict[str, ClaimNode]
    ) -> tuple[str, str]:
        """Generate cache key for edge typing."""
        src_hash = node_map.get(src_id, ClaimNode("", "", "", "", "", 0, "")).text_hash
        dst_hash = node_map.get(dst_id, ClaimNode("", "", "", "", "", 0, "")).text_hash
        # Normalize order for bidirectional caching
        return tuple(sorted([src_hash, dst_hash]))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Quality Gate
    # ─────────────────────────────────────────────────────────────────────────
    
    def _quality_gate(
        self, 
        num_candidates: int,
        candidates: list[CandidateEdge],
        kept_edges: list[TypedEdge],
        result: GraphResult,  # M75: Pass result to update metrics
    ) -> bool:
        """
        Check if kept_ratio is within acceptable bounds.
        
        M75: Exclude cross-topic edges from the ratio calculation.
        """
        # Identify cross-topic pairs
        cross_topic_pairs = {
            (c.src_id, c.dst_id) 
            for c in candidates 
            if c.cross_topic
        }
        
        # Filter numerator: kept edges that are NOT cross-topic
        kept_within = [
            e for e in kept_edges 
            if (e.src_id, e.dst_id) not in cross_topic_pairs
        ]
        
        # Filter denominator: candidates that are NOT cross-topic
        cand_within = [c for c in candidates if not c.cross_topic]
        
        # M75: Track metrics
        result.within_topic_edges_count = len(cand_within)
        result.cross_topic_edges_count = len(candidates) - len(cand_within)
        
        if not cand_within:
            # If no within-topic candidates (rare), we can't calculate valid ratio.
            # Fallback to passing if we have specific cross-topic edges or fail?
            # Safe default: if we have NO within-topic candidates, maybe the whole graph is cross-topic?
            # Let's use overall ratio as fallback.
            numerator = len(kept_edges)
            denominator = len(candidates)
            logger.debug("[M75] No within-topic candidates, using overall ratio")
        else:
            numerator = len(kept_within)
            denominator = len(cand_within)
            
        kept_ratio = numerator / denominator if denominator > 0 else 0
        result.kept_ratio_within_topic = kept_ratio
        
        # Check constraints
        if kept_ratio < self.config.min_kept_ratio:
            logger.warning(
                "[M72] Quality gate: kept_ratio %.3f < min %.3f (within-focus)",
                kept_ratio, self.config.min_kept_ratio
            )
            return False
        
        if kept_ratio > self.config.max_kept_ratio:
            logger.warning(
                "[M72] Quality gate: kept_ratio %.3f > max %.3f (within-focus)",
                kept_ratio, self.config.max_kept_ratio
            )
            return False
        
        return True
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 8: Graph Ranking (PageRank)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _rank_claims(
        self,
        nodes: list[ClaimNode],
        edges: list[TypedEdge],
    ) -> list[RankedClaim]:
        """
        Rank claims using PageRank and structural weight.
        
        Uses networkx for PageRank computation.
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("[M72] networkx not installed, skipping ranking")
            return [
                RankedClaim(
                    claim_id=n.claim_id,
                    centrality_score=n.importance,
                    in_structural_weight=0.0,
                    in_contradict_weight=0.0,
                    is_key_claim=i < self.config.top_k,
                )
                for i, n in enumerate(sorted(nodes, key=lambda x: x.importance, reverse=True))
            ]
        
        # Build directed weighted graph
        G = nx.DiGraph()
        
        # Add all nodes
        for node in nodes:
            G.add_node(node.claim_id, importance=node.importance)
        
        # Add edges with weights
        for edge in edges:
            weight = edge.score * RELATION_MULTIPLIERS.get(edge.relation, 0.0)
            if weight > 0:
                G.add_edge(edge.src_id, edge.dst_id, weight=weight, relation=edge.relation)
        
        # Compute PageRank
        try:
            pagerank = nx.pagerank(G, weight="weight", alpha=0.85)
        except Exception as e:
            logger.warning("[M72] PageRank failed: %s", e)
            pagerank = {n.claim_id: n.importance for n in nodes}
        
        # Calculate structural and contradiction weights
        structural_weights: dict[str, float] = defaultdict(float)
        contradict_weights: dict[str, float] = defaultdict(float)
        
        for edge in edges:
            weight = edge.score * RELATION_MULTIPLIERS.get(edge.relation, 0.0)
            
            if edge.relation in STRUCTURAL_RELATIONS:
                structural_weights[edge.dst_id] += weight
            elif edge.relation == EdgeRelation.CONTRADICTS:
                contradict_weights[edge.dst_id] += weight
        
        # Create ranked claims
        ranked: list[RankedClaim] = []
        for node in nodes:
            ranked.append(RankedClaim(
                claim_id=node.claim_id,
                centrality_score=pagerank.get(node.claim_id, 0.0),
                in_structural_weight=structural_weights.get(node.claim_id, 0.0),
                in_contradict_weight=contradict_weights.get(node.claim_id, 0.0),
                is_key_claim=False,  # Will be set below
            ))
        
        # Combined score: centrality + structural weight (normalized)
        max_centrality = max((r.centrality_score for r in ranked), default=1.0) or 1.0
        max_structural = max((r.in_structural_weight for r in ranked), default=1.0) or 1.0
        
        for r in ranked:
            r._combined_score = (
                r.centrality_score / max_centrality * 0.6 +
                r.in_structural_weight / max_structural * 0.4
            )
        
        # Sort by combined score and select top-K as key claims
        ranked.sort(key=lambda r: r._combined_score, reverse=True)
        
        for i, r in enumerate(ranked):
            r.is_key_claim = i < self.config.top_k
            # Clean up temporary attribute
            if hasattr(r, "_combined_score"):
                delattr(r, "_combined_score")
        
        logger.debug("[M72] Ranking: %d claims, top %d selected as key",
                    len(ranked), min(len(ranked), self.config.top_k))
        
        return ranked
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.embedding_client.clear_cache()
        self._edge_cache.clear()
