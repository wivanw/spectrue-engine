# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Phase Runner

Executes progressive widening phases with early exit on sufficiency.

Waterfall Execution:
1. For each claim, execute phases A → B → C → D sequentially
2. After each phase, check sufficiency
3. If sufficient: stop early, skip remaining phases
4. If insufficient: continue to next phase

Parallelism:
- Phase execution is parallelized ACROSS claims (within same phase)
- Phase progression is sequential (all claims finish Phase A before B)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spectrue_core.verification.orchestration.execution_plan import (
    Phase,
    ExecutionPlan,
    ExecutionState,
)
from spectrue_core.verification.orchestration.sufficiency import (
    check_sufficiency_for_claim,
    judge_sufficiency_for_claim,
    verdict_ready_for_claim,
    SufficiencyDecision,
    SufficiencyStatus,
    get_domain_tier,  # returns EvidenceChannel (enum)
)
from spectrue_core.agents.skills.query import generate_followup_query_from_evidence
from spectrue_core.utils.trace import Trace
from spectrue_core.tools.trusted_sources import get_trusted_domains_by_lang
from spectrue_core.schema.claim_metadata import EvidenceChannel
from spectrue_core.verification.search.source_utils import canonicalize_sources, extract_domain
from spectrue_core.verification.search.retrieval_eval import evaluate_retrieval_confidence
from spectrue_core.verification.orchestration.stop_decision import EVStopParams, evaluate_stop_decision
from spectrue_core.verification.search.search_escalation import (
    build_query_variants,
    select_topic_from_claim,
    compute_retrieval_outcome,
    should_stop_escalation,
    get_escalation_ladder,
    compute_escalation_reason_codes,
    trace_query_variants,
    trace_topic_selected,
    trace_escalation_pass,
    trace_search_stop,
    trace_search_summary,
    EscalationConfig,
    QueryVariant,
    RetrievalOutcome,
)

if TYPE_CHECKING:
    from spectrue_core.verification.search.search_mgr import SearchManager
    from spectrue_core.verification.evidence.evidence_pack import Claim

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase Search Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseSearchResult:
    """Result from executing a single phase for a claim."""
    claim_id: str
    phase_id: str
    sources: list[dict]
    error: str | None = None


@dataclass
class RetrievalHop:
    """Minimal hop state for iterative retrieval."""
    hop_index: int
    query: str
    decision: SufficiencyDecision
    reason: str
    phase_id: str | None = None
    query_type: str | None = None
    results: list[dict] | None = None
    retrieval_eval: dict | None = None

    def to_dict(self) -> dict:
        return {
            "hop_index": int(self.hop_index),
            "query": self.query,
            "decision": self.decision.value,
            "reason": self.reason,
            "phase_id": self.phase_id,
            "query_type": self.query_type,
            "results_count": len(self.results or []),
            "retrieval_eval": self.retrieval_eval or {},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Phase Runner
# ─────────────────────────────────────────────────────────────────────────────

class PhaseRunner:
    """
    Executes progressive widening phases with early exit.
    
    Example:
        runner = PhaseRunner(search_mgr)
        evidence = await runner.run_all_claims(claims, execution_plan)
    """

    def __init__(
        self,
        search_mgr: SearchManager,
        *,
        max_concurrent: int = 3,
        progress_callback=None,
        use_retrieval_loop: bool = False,
        policy_profile=None,
        can_add_search=None,
        max_cost: int | None = None,
        inline_sources: list[dict] | None = None,
        agent: Any | None = None,
        ev_stop_params: EVStopParams | None = None,
    ) -> None:
        """
        Initialize PhaseRunner.
        
        Args:
            search_mgr: SearchManager for executing searches
            max_concurrent: Maximum concurrent searches per phase
            progress_callback: Optional async callback for progress updates
            inline_sources: Pre-verified inline sources to include in evidence
            agent: Agent instance for verification (M109)
        """
        self.search_mgr = search_mgr
        self.agent = agent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.execution_state = ExecutionState()
        self.progress_callback = progress_callback
        self.use_retrieval_loop = use_retrieval_loop
        self.policy_profile = policy_profile
        self.can_add_search = can_add_search
        self.max_cost = max_cost
        self.inline_sources = inline_sources or []
        self.ev_stop_params = ev_stop_params
        """Optional value-based stop parameters (M113)."""
        if self.inline_sources:
            import logging
            logging.getLogger(__name__).debug(
                "PhaseRunner received %d inline sources", len(self.inline_sources)
            )

    async def run_all_claims(
        self,
        claims: list[Claim],
        execution_plan: ExecutionPlan,
    ) -> dict[str, list[dict]]:
        """
        Execute all phases for all claims with progressive widening.
        
        Uses waterfall pattern:
        - All claims execute Phase A in parallel
        - Check sufficiency for each claim
        - Claims that are sufficient skip later phases
        - Continue to Phase B, C, D for remaining claims
        
        Args:
            claims: List of claims to search for
            execution_plan: ExecutionPlan with phases per claim
            
        Returns:
            Dict mapping claim_id -> list of sources
        """
        # Initialize state and evidence
        claim_map = {c.get("id"): c for c in claims}
        evidence: dict[str, list[dict]] = {c.get("id"): [] for c in claims}

        if self.use_retrieval_loop:
            evidence = await self._run_claims_with_loop(
                claims=claims,
                execution_plan=execution_plan,
            )
            self._emit_claim_trace_summaries(execution_plan)
            return evidence

        # === PRE-FETCH: Batch extract all inline source URLs ===
        # This populates the cache so per-claim EAL calls get cache hits
        await self._pre_extract_inline_urls()

        dependency_layers = self._get_dependency_layers(claims)

        # Determine phase order (collect all unique phases in order)
        phase_order = ["A-light", "A", "A-origin", "B", "C", "D"]

        for phase_id in phase_order:
            for layer in dependency_layers:
                # Get claims that need this phase (and aren't already sufficient)
                claims_for_phase = self._get_claims_needing_phase(
                    claims=layer,
                    execution_plan=execution_plan,
                    phase_id=phase_id,
                )

                if not claims_for_phase:
                    continue

                logger.debug(
                    "[Orchestration] Running Phase %s for %d claim(s)",
                    phase_id, len(claims_for_phase)
                )

                # Emit progress: searching_phase_X
                if self.progress_callback:
                    status_key = f"searching_phase_{phase_id.lower().replace('-', '_')}"
                    try:
                        await self.progress_callback(status_key)
                    except Exception as e:
                        logger.warning("[Orchestration] Progress callback failed: %s", e)

                # Execute phase in parallel for all eligible claims
                phase_results = await self._run_phase_parallel(
                    claims=claims_for_phase,
                    execution_plan=execution_plan,
                    phase_id=phase_id,
                )

                # Process results and check sufficiency
                for result in phase_results:
                    claim_id = result.claim_id
                    claim_state = self.execution_state.get_or_create(claim_id)

                    if result.error:
                        claim_state.error = result.error
                        Trace.event("phase.error", {
                            "claim_id": claim_id,
                            "phase_id": result.phase_id,
                            "error": result.error,
                        })
                        continue

                    # Accumulate sources
                    evidence[claim_id].extend(result.sources)
                    claim_state.mark_completed(result.phase_id)

                    # T20: Trace phase completion
                    Trace.event("phase.completed", {
                        "claim_id": claim_id,
                        "phase_id": result.phase_id,
                        "results_count": len(result.sources),
                        "total_sources": len(evidence[claim_id]),
                    })

                    # Check sufficiency
                    claim = claim_map.get(claim_id)
                    if claim and isinstance(claim, dict):
                        sufficiency = check_sufficiency_for_claim(
                            claim=claim,
                            sources=evidence[claim_id],
                        )

                        if sufficiency.status == SufficiencyStatus.SUFFICIENT:
                            # Determine remaining phases for this claim
                            claim_phases = execution_plan.get_phases(claim_id)
                            current_idx = next(
                                (i for i, p in enumerate(claim_phases) if p.phase_id == result.phase_id),
                                -1
                            )
                            remaining = [p.phase_id for p in claim_phases[current_idx + 1:]]

                            claim_state.mark_sufficient(
                                reason=sufficiency.reason,
                                remaining_phases=remaining,
                            )

                            # T20: Trace early exit
                            Trace.event("phase.stopped", {
                                "claim_id": claim_id,
                                "phase_id": result.phase_id,
                                "reason": "sufficiency_met",
                                "rule": sufficiency.rule_matched,
                                "skipped_phases": remaining,
                            })

                            logger.debug(
                                "[Orchestration] Claim %s sufficient after Phase %s (rule=%s), skipping %s",
                                claim_id, result.phase_id, sufficiency.rule_matched, remaining
                            )
                        elif sufficiency.status == SufficiencyStatus.SKIP:
                            claim_phases = execution_plan.get_phases(claim_id)
                            current_idx = next(
                                (i for i, p in enumerate(claim_phases) if p.phase_id == result.phase_id),
                                -1,
                            )
                            remaining = [p.phase_id for p in claim_phases[current_idx + 1:]]
                            claim_state.mark_sufficient(
                                reason=sufficiency.reason,
                                remaining_phases=remaining,
                            )
                            Trace.event("phase.stopped", {
                                "claim_id": claim_id,
                                "phase_id": result.phase_id,
                                "reason": "sufficiency_skip",
                                "rule": sufficiency.rule_matched,
                                "skipped_phases": remaining,
                            })
                            logger.debug(
                                "[Orchestration] Claim %s skipped after Phase %s (reason=%s), skipping %s",
                                claim_id, result.phase_id, sufficiency.reason, remaining
                            )
                        else:
                            # T20: Trace continue
                            Trace.event("phase.continue", {
                                "claim_id": claim_id,
                                "phase_id": result.phase_id,
                                "reason": sufficiency.reason,
                            })

        # === CENTRALIZED BATCH ENRICHMENT ===
        # After all searches complete, enrich all sources at once
        evidence = await self._batch_enrich_all_sources(evidence)

        self._emit_claim_trace_summaries(execution_plan)
        return evidence

    def _emit_claim_trace_summaries(self, execution_plan: ExecutionPlan) -> None:
        summaries = self.execution_state.build_trace_summaries(plan=execution_plan)
        for summary in summaries:
            Trace.event("claim.trace.summary", summary)

    async def _pre_extract_inline_urls(self) -> None:
        """
        Pre-fetch all inline source URLs to populate cache.
        
        This ensures subsequent EAL calls get cache hits instead of
        making individual API calls (5 URLs = 1 credit vs 1 URL = 1 credit).
        """
        if not self.inline_sources:
            return
        
        # Collect all URLs from inline sources
        all_urls: list[str] = []
        for src in self.inline_sources:
            if not isinstance(src, dict):
                continue
            url = src.get("url") or src.get("link")
            if url and isinstance(url, str) and url not in all_urls:
                all_urls.append(url)
        
        if not all_urls:
            return
        
        # Batch extract using search_mgr
        if hasattr(self.search_mgr, "fetch_urls_content_batch"):
            Trace.event("phase_runner.prefetch.start", {
                "total_urls": len(all_urls),
                "source": "inline_sources",
            })
            try:
                await self.search_mgr.fetch_urls_content_batch(all_urls)
                Trace.event("phase_runner.prefetch.done", {
                    "urls_extracted": len(all_urls),
                })
            except Exception as e:
                logger.warning("[PhaseRunner] Pre-fetch failed: %s", e)
                Trace.event("phase_runner.prefetch.error", {
                    "error": str(e)[:200],
                })

    async def _batch_enrich_all_sources(
        self, evidence: dict[str, list[dict]]
    ) -> dict[str, list[dict]]:
        """
        Centralized batch enrichment for all sources across all claims.
        
        Collects all URLs from:
        1. All claim sources (from search results)
        2. Inline sources (article citations)
        
        Fetches them in batch (5 URLs = 1 credit), then applies content.
        
        This consolidates ALL URL extraction for ~90% cost reduction.
        """
        # Collect all URLs from all claims + inline sources
        all_urls: list[str] = []
        url_to_sources: dict[str, list[dict]] = {}  # Track which sources need each URL
        
        # 1. Collect from claim sources
        for claim_id, sources in evidence.items():
            for src in sources:
                if not isinstance(src, dict):
                    continue
                url = src.get("url") or src.get("link")
                if url and isinstance(url, str):
                    if url not in all_urls:
                        all_urls.append(url)
                    url_to_sources.setdefault(url, []).append(src)
        
        # 2. Collect from inline sources
        for src in self.inline_sources:
            if not isinstance(src, dict):
                continue
            url = src.get("url") or src.get("link")
            if url and isinstance(url, str):
                if url not in all_urls:
                    all_urls.append(url)
                url_to_sources.setdefault(url, []).append(src)
        
        if not all_urls:
            Trace.event("orchestration.batch_enrich.skip", {
                "reason": "no_urls",
            })
            return evidence
        
        # Batch fetch all URLs at once
        if not hasattr(self.search_mgr, "fetch_urls_content_batch"):
            logger.warning("[PhaseRunner] SearchManager lacks fetch_urls_content_batch")
            return evidence
        
        Trace.event("orchestration.batch_enrich.start", {
            "total_urls": len(all_urls),
            "claims_count": len(evidence),
            "inline_count": len(self.inline_sources),
        })
        
        try:
            content_map = await self.search_mgr.fetch_urls_content_batch(all_urls)
            
            # Apply content to ALL sources that reference each URL
            enriched_count = 0
            for url, content in content_map.items():
                for src in url_to_sources.get(url, []):
                    if not src.get("fulltext"):
                        src["content"] = content
                        src["fulltext"] = True
                        enriched_count += 1
            
            Trace.event("orchestration.batch_enrich.done", {
                "urls_fetched": len(content_map),
                "sources_enriched": enriched_count,
                "batch_efficiency": f"{len(all_urls)}/{(len(all_urls) + 4) // 5}",
            })
            
        except Exception as e:
            logger.warning("[PhaseRunner] Batch enrichment failed: %s", e)
            Trace.event("orchestration.batch_enrich.error", {
                "error": str(e)[:200],
            })
        
        return evidence

    async def _run_claims_with_loop(
        self,
        *,
        claims: list[Claim],
        execution_plan: ExecutionPlan,
    ) -> dict[str, list[dict]]:
        """
        Run bounded retrieval loops per claim (replaces linear waterfall).
        """
        evidence: dict[str, list[dict]] = {c.get("id"): [] for c in claims}
        dependency_layers = self._get_dependency_layers(claims)

        # === PRE-FETCH: Batch extract all inline source URLs ===
        # This populates the cache so EAL calls get cache hits
        await self._pre_extract_inline_urls()

        for layer in dependency_layers:
            tasks = []
            for claim in layer:
                claim_id = claim.get("id")
                phases = execution_plan.get_phases(claim_id)
                tasks.append(self._run_retrieval_loop_for_claim(claim, phases))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(results):
                claim_id = layer[idx].get("id", "unknown")
                state = self.execution_state.get_or_create(claim_id)
                if isinstance(result, Exception):
                    state.error = str(result)
                    continue

                sources, hops, decision, reason = result
                evidence[claim_id] = sources
                state.hops = list(hops)

                for hop in hops:
                    if hop.phase_id:
                        state.mark_completed(hop.phase_id)

                if decision == SufficiencyDecision.ENOUGH:
                    remaining = []
                    if hops:
                        last_phase = hops[-1].phase_id
                        claim_phases = execution_plan.get_phases(claim_id)
                        if last_phase:
                            last_idx = next(
                                (i for i, p in enumerate(claim_phases) if p.phase_id == last_phase),
                                -1,
                            )
                            remaining = [p.phase_id for p in claim_phases[last_idx + 1:]]
                    state.mark_sufficient(reason=reason, remaining_phases=remaining)
                elif decision == SufficiencyDecision.STOP:
                    state.sufficiency_reason = reason
                    state.stop_reason = reason  # Ensure stop_reason is set for terminal states

        return evidence

    def _get_dependency_layers(self, claims: list[Claim]) -> list[list[Claim]]:
        """
        Build dependency-ordered layers of claims.

        Claims with dependencies are placed after their prerequisites.
        Claims without dependencies remain in the first available layer.
        """
        claim_map: dict[str, Claim] = {}
        claim_order: list[str] = []
        no_id_claims: list[Claim] = []

        for claim in claims:
            claim_id = claim.get("id")
            if not claim_id:
                no_id_claims.append(claim)
                continue
            claim_map[claim_id] = claim
            claim_order.append(claim_id)

        deps_by_id: dict[str, list[str]] = {cid: [] for cid in claim_order}
        has_deps = False
        for cid in claim_order:
            claim = claim_map.get(cid, {})
            structure = claim.get("structure")
            if not isinstance(structure, dict):
                continue
            deps_raw = structure.get("dependencies", [])
            if not isinstance(deps_raw, list):
                continue
            deps = [
                d for d in deps_raw
                if isinstance(d, str) and d in claim_map and d != cid
            ]
            if deps:
                has_deps = True
            deps_by_id[cid] = deps

        if not has_deps:
            return [claims]

        indegree: dict[str, int] = {cid: 0 for cid in claim_order}
        outgoing: dict[str, list[str]] = {cid: [] for cid in claim_order}
        for cid, deps in deps_by_id.items():
            for dep in deps:
                outgoing.setdefault(dep, []).append(cid)
                indegree[cid] += 1

        layers: list[list[Claim]] = []
        visited: set[str] = set()
        available = [cid for cid in claim_order if indegree[cid] == 0]

        while available:
            layer_ids = [cid for cid in claim_order if cid in available]
            layer_claims = [claim_map[cid] for cid in layer_ids]
            layers.append(layer_claims)

            for cid in layer_ids:
                visited.add(cid)
                for nxt in outgoing.get(cid, []):
                    indegree[nxt] -= 1

            available = [
                cid for cid in claim_order
                if cid not in visited and indegree[cid] == 0
            ]

        if len(visited) != len(claim_order):
            remaining = [cid for cid in claim_order if cid not in visited]
            logger.warning(
                "[M93] Dependency cycle detected among claims: %s",
                ", ".join(remaining),
            )
            layers.append([claim_map[cid] for cid in remaining])

        if no_id_claims:
            if not layers:
                layers = [list(no_id_claims)]
            else:
                layers[0].extend(no_id_claims)

        return layers

    async def _run_retrieval_loop_for_claim(
        self,
        claim: Claim,
        phases: list[Phase],
    ) -> tuple[list[dict], list[RetrievalHop], SufficiencyDecision, str]:
        """
        Execute bounded iterative retrieval for a single claim.
        """
        claim_id = claim.get("id", "unknown")
        claim_text = claim.get("normalized_text", "") or claim.get("text", "")
        all_sources: list[dict] = []
        hops: list[RetrievalHop] = []

        # Create claim-specific copies of inline sources to avoid mutation
        # Each claim needs its own copy with claim_id set correctly
        claim_inline_sources: list[dict] = []
        for src in self.inline_sources:
            if isinstance(src, dict):
                src_copy = dict(src)
                src_copy["claim_id"] = claim_id
                claim_inline_sources.append(src_copy)


        # CRITICAL SHORTCUT: Check inline sources first (M109)
        if claim_inline_sources:
            logger.debug("Checking inline sources shortcut for claim %s", claim_id)

            # Enrich inline sources if agent is available
            if self.agent and hasattr(self.agent, "verify_inline_source_relevance"):
                for src in claim_inline_sources:
                    if not src.get("quote_matches"):
                        try:
                            # verify_inline_source_relevance expects: claims (list[dict]), inline_source (dict), article_excerpt
                            verification = await self.agent.verify_inline_source_relevance(
                                claims=[claim],  # Pass claim as list
                                inline_source=src,
                                article_excerpt=claim_text[:500],
                            )
                            if verification:
                                src.update(verification)
                        except Exception as e:
                            logger.warning("Inline verification failed for %s: %s", src.get("url"), e)

            sufficient, stats = verdict_ready_for_claim(
                claim_inline_sources,
                claim_id=claim_id,
            )

            if sufficient:
                logger.info("Inline sources sufficient (shortcut) for %s. Stats: %s", claim_id, stats)
                return list(claim_inline_sources), hops, SufficiencyDecision.ENOUGH, "inline_sufficient"

        if not phases:
            return all_sources, hops, SufficiencyDecision.STOP, "no_phases"

        max_hops = 1
        if self.policy_profile is not None:
            max_hops = self.policy_profile.stop_conditions.max_hops or 1
        else:
            max_hops = max(len(phases), 1)

        next_query = None
        next_query_type = None
        decision = SufficiencyDecision.NEED_FOLLOWUP
        reason = ""

        for hop_index in range(max_hops):
            if not self._budget_allows_hop():
                decision = SufficiencyDecision.STOP
                reason = "budget_ceiling"
                break

            phase = phases[min(hop_index, len(phases) - 1)] if phases else None
            phase_id = phase.phase_id if phase else None

            query = next_query or self._select_claim_query(claim)
            if not query:
                decision = SufficiencyDecision.STOP
                reason = "missing_query"
                break

            Trace.event("retrieval.hop.started", {
                "claim_id": claim_id,
                "hop": hop_index,
                "phase_id": phase_id,
                "query_type": next_query_type or "initial",
            })

            async with self.semaphore:
                sources = await self._search_by_phase(claim, phase, query_override=query)

            for src in sources:
                if isinstance(src, dict):
                    if not src.get("claim_id"):
                        src["claim_id"] = claim_id
                    if claim_text and not src.get("claim_text"):
                        src["claim_text"] = claim_text

            # NOTE: Content enrichment deferred to _batch_enrich_all_sources after
            # all searches complete. This enables batching (5 URLs = 1 credit).
            # Sources collected here contain only snippets from search results.

            all_sources.extend(sources)

            expected_cost = None
            try:
                expected_cost = self.search_mgr.estimate_hop_cost()
            except Exception:
                expected_cost = None
            evaluation = evaluate_retrieval_confidence(
                sources,
                runtime_config=getattr(getattr(self.search_mgr, "config", None), "runtime", None),
                expected_cost=expected_cost,
            )

            # Value-based stop decision (EV) if enabled.
            if self.ev_stop_params is not None:
                try:
                    stop_result = evaluate_stop_decision(
                        posterior_true=float(evaluation.get("retrieval_confidence", 0.0)),
                        expected_delta_p=float(evaluation.get("expected_gain", 0.0)),
                        params=self.ev_stop_params,
                        budget_remaining=None,
                        quality_signal=evaluation,
                    )
                    Trace.event(
                        "retrieval.stop_decision",
                        {
                            "claim_id": claim_id,
                            "hop": hop_index,
                            "phase_id": phase_id,
                            **stop_result.to_dict(),
                        },
                    )
                    if stop_result.should_stop:
                        decision = SufficiencyDecision.STOP
                        reason = f"ev_stop:{stop_result.reason}"
                        break
                except Exception as e:
                    Trace.event(
                        "retrieval.stop_decision.error",
                        {
                            "claim_id": claim_id,
                            "hop": hop_index,
                            "phase_id": phase_id,
                            "error": str(e)[:200],
                        },
                    )

            action_result = None
            decide_fn = getattr(self.search_mgr, "decide_retrieval_action", None)
            if callable(decide_fn):
                try:
                    action_result = decide_fn(retrieval_eval=evaluation, claim=claim)
                except Exception:
                    action_result = None

            if isinstance(action_result, tuple) and len(action_result) == 2:
                action, action_reason = action_result
            else:
                action, action_reason = "continue", "decision_default"
            allowed_actions = {
                "stop_early",
                "continue",
                "refine_query",
                "change_language",
                "restrict_domains",
                "change_channel",
            }
            if action not in allowed_actions:
                action = "continue"
                action_reason = "decision_default"
            Trace.event(
                "retrieval.evaluation",
                {
                    "claim_id": claim_id,
                    "hop": hop_index,
                    "scores": evaluation,
                    "action": action,
                    "action_reason": action_reason,
                },
            )

            judge = judge_sufficiency_for_claim(
                claim=claim,
                sources=all_sources,
                policy_profile=self.policy_profile,
                use_policy_by_channel=getattr(phase, "use_policy_by_channel", {}),
            )

            decision = judge.decision
            reason = judge.reason

            claim_sources = [s for s in all_sources if isinstance(s, dict) and s.get("claim_id") == claim_id]

            # Include claim-specific inline sources in potential evidence
            # claim_inline_sources already have claim_id set correctly
            potential_evidence = claim_sources + claim_inline_sources
            logger.debug(
                "verdict_ready call: claim_id=%s sources=%d (claim=%d + inline=%d)",
                claim_id, len(potential_evidence), len(claim_sources), len(claim_inline_sources)
            )
            ready, ready_stats = verdict_ready_for_claim(
                potential_evidence,
                claim_id=str(claim_id),
            )

            if ready:
                logger.info(
                    "Initial sufficiency check passed for claim %s (stats=%s)",
                    claim_id, ready_stats
                )

                # NOTE: Inline source enrichment deferred to _batch_enrich_all_sources
                # to enable batched URL extraction (5 URLs = 1 credit).

                # IMPORTANT: We must return the inline sources too!
                # Filter to only unique sources to avoid dupes if they were already in claim_sources
                seen_urls = {s.get("url") for s in claim_sources if s.get("url")}
                combined = list(claim_sources)
                for src in claim_inline_sources:
                    if src.get("url") not in seen_urls:
                        combined.append(src)

                return combined, hops, SufficiencyDecision.ENOUGH, "inline_sufficient"

            # ─────────────────────────────────────────────────────────────────────────────
            # Retrieval Loop
            # ─────────────────────────────────────────────────────────────────────────────
            # Respect retrieval action from confidence evaluation
            # Only sync action with decision - don't blindly override stop_early
            if decision == SufficiencyDecision.ENOUGH:
                action = "stop_early"
                action_reason = "evidence_sufficient"
            elif decision == SufficiencyDecision.NEED_FOLLOWUP:
                # Sufficiency explicitly needs more evidence - continue even if confidence was high
                if action == "stop_early":
                    action = "continue"
                    action_reason = "sufficiency_requires_more"

            hop_state = RetrievalHop(
                hop_index=hop_index,
                query=query,
                decision=decision,
                reason=reason,
                phase_id=phase_id,
                query_type=next_query_type,
                results=sources,
                retrieval_eval={
                    **evaluation,
                    "action": action,
                    "action_reason": action_reason,
                },
            )
            hops.append(hop_state)

            Trace.event("retrieval.hop.completed", {
                "claim_id": claim_id,
                "hop": hop_index,
                "decision": decision.value,
                "reason": reason,
                "results_count": len(sources),
            })

            if decision == SufficiencyDecision.ENOUGH:
                break
            if decision == SufficiencyDecision.STOP:
                break

            if hop_index >= max_hops - 1:
                decision = SufficiencyDecision.STOP
                reason = "max_hops_reached"
                Trace.event("retrieval.max_hops_reached", {
                    "claim_id": claim_id,
                    "max_hops": max_hops,
                    "hops_completed": len(hops),
                })
                break

            if action in ("refine_query", "change_language", "restrict_domains", "change_channel"):
                next_query_type = action
                reason = action_reason

            followup = generate_followup_query_from_evidence(claim_text, sources)
            if not followup:
                decision = SufficiencyDecision.STOP
                reason = "followup_failed"
                break

            next_query = followup["query"]
            next_query_type = followup["query_type"]

        return all_sources, hops, decision, reason

    def _budget_allows_hop(self) -> bool:
        if not self.can_add_search:
            return True
        # Assume "smart" or equivalent for search type placeholder
        return bool(self.can_add_search("gpt-5-nano", "smart", self.max_cost))

    def _get_claims_needing_phase(
        self,
        claims: list[Claim],
        execution_plan: ExecutionPlan,
        phase_id: str,
    ) -> list[Claim]:
        """
        Get claims that:
        1. Have this phase in their plan
        2. Are not already sufficient
        3. Haven't errored out
        """
        result = []

        for claim in claims:
            claim_id = claim.get("id")

            # Check if claim has this phase
            claim_phases = execution_plan.get_phases(claim_id)
            has_phase = any(p.phase_id == phase_id for p in claim_phases)
            if not has_phase:
                continue

            # Check if already sufficient or errored
            state = self.execution_state.get_or_create(claim_id)
            if state.is_sufficient or state.error:
                continue

            result.append(claim)

        return result

    async def _run_phase_parallel(
        self,
        claims: list[Claim],
        execution_plan: ExecutionPlan,
        phase_id: str,
    ) -> list[PhaseSearchResult]:
        """
        Run a phase for multiple claims in parallel (with semaphore).
        """
        tasks = []

        for claim in claims:
            claim_id = claim.get("id")
            claim_phases = execution_plan.get_phases(claim_id)
            phase = next((p for p in claim_phases if p.phase_id == phase_id), None)

            if phase:
                task = self._run_phase_for_claim(claim, phase)
                tasks.append(task)

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed: list[PhaseSearchResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                claim_id = claims[i].get("id", "unknown") if i < len(claims) else "unknown"
                processed.append(PhaseSearchResult(
                    claim_id=claim_id,
                    phase_id=phase_id,
                    sources=[],
                    error=str(r),
                ))
            elif isinstance(r, PhaseSearchResult):
                processed.append(r)

        return processed

    async def _run_phase_for_claim(
        self,
        claim: Claim,
        phase: Phase,
    ) -> PhaseSearchResult:
        """
        Execute a single phase for a single claim.
        
        Maps Phase configuration to search parameters.
        """
        claim_id = claim.get("id", "unknown")

        # T20: Trace phase start
        Trace.event("phase.started", {
            "claim_id": claim_id,
            "phase_id": phase.phase_id,
            "locale": phase.locale,
            "channels": [c.value for c in phase.channels],
            "max_results": phase.max_results,
        })

        async with self.semaphore:
            try:
                sources = await self._search_by_phase(claim, phase)

                return PhaseSearchResult(
                    claim_id=claim_id,
                    phase_id=phase.phase_id,
                    sources=sources,
                )

            except Exception as e:
                # T27: Fail-soft - log warning and return empty results
                logger.warning(
                    "[Orchestration] Phase %s failed for claim %s: %s",
                    phase.phase_id, claim_id, e
                )
                # T27: Trace search failure
                Trace.event("search.failed", {
                    "claim_id": claim_id,
                    "phase_id": phase.phase_id,
                    "error_type": type(e).__name__,
                    "error": str(e)[:200],  # Truncate for safe logging
                })
                return PhaseSearchResult(
                    claim_id=claim_id,
                    phase_id=phase.phase_id,
                    sources=[],
                    error=str(e),
                )

    def _select_claim_query(self, claim: Claim) -> str:
        """Select query for claim search, with deterministic fallback."""
        query_source = "claim_search_queries"
        queries = claim.get("search_queries", [])
        
        if not queries:
            candidates = claim.get("query_candidates", [])
            if candidates:
                queries = [c.get("text") for c in candidates if c.get("text")]
                query_source = "phase_candidates"
        
        if queries:
            query = queries[0]
        else:
            # Deterministic keyword extraction fallback (NO normalized_text)
            claim_text = claim.get("text", "")
            query = self._extract_keywords_fallback(claim_text)
            query_source = "keyword_fallback" if query != claim_text[:80] else "text_truncate"
        
        # Trace event for debugging query selection
        Trace.event("search.query.selected", {
            "claim_id": claim.get("id", "unknown"),
            "query": query[:100] if query else "",
            "query_source": query_source,
            "topic": claim.get("search_method", "news"),
        })
        
        return query if query else ""
    
    def _extract_keywords_fallback(self, text: str, max_tokens: int = 6) -> str:
        """Extract keyword query from text (deterministic, language-agnostic)."""
        import re
        if not text:
            return ""
        lowered = text.lower()
        cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
        tokens = cleaned.split()
        unique_tokens: list[str] = []
        seen: set[str] = set()
        for t in tokens:
            if len(t) >= 3 and t not in seen:
                unique_tokens.append(t)
                seen.add(t)
            if len(unique_tokens) >= max_tokens:
                break
        if unique_tokens:
            return " ".join(unique_tokens)
        return text[:80].rstrip(".,!?;")

    async def _run_escalation_ladder(
        self,
        claim: Claim,
        phase: Phase | None,
        escalation_config: EscalationConfig | None = None,
    ) -> tuple[list[dict], RetrievalOutcome, int, int, bool]:
        """
        Run multi-pass escalation ladder for a claim.
        
        Tries query variants with escalating parameters until evidence found
        or ladder exhausted.
        
        Args:
            claim: The claim to search for
            phase: Phase configuration (for locale/channels)
            escalation_config: Optional escalation thresholds
            
        Returns:
            (sources, final_outcome, passes_executed, tavily_calls, domains_relaxed)
        """
        claim_id = claim.get("id", "unknown")
        config = escalation_config or EscalationConfig()
        
        # Build query variants from structured claim fields
        query_variants = build_query_variants(claim, config)
        if not query_variants:
            # Fallback to legacy query selection if no variants
            fallback_query = self._select_claim_query(claim)
            if fallback_query:
                query_variants = [QueryVariant(
                    query_id="Q_fallback",
                    text=fallback_query,
                    strategy="legacy_fallback",
                )]
        
        trace_query_variants(claim_id, query_variants)
        
        if not query_variants:
            return [], RetrievalOutcome(0, 0.0, 0, 0), 0, 0, False
        
        # Build query lookup
        query_by_id: dict[str, str] = {v.query_id: v.text for v in query_variants}
        
        # Get topic from structured fields
        topic, topic_reasons = select_topic_from_claim(claim)
        trace_topic_selected(claim_id, topic, topic_reasons)
        
        # Get escalation ladder
        ladder = get_escalation_ladder()
        
        all_sources: list[dict] = []
        passes_executed = 0
        tavily_calls = 0
        domains_relaxed = False
        reason_codes: list[str] = ["initial"]
        
        for pass_config in ladder:
            passes_executed += 1
            pass_sources: list[dict] = []
            
            for query_id in pass_config.query_ids:
                query_text = query_by_id.get(query_id)
                if not query_text:
                    continue
                
                # Determine include_domains
                include_domains: list[str] | None = None
                if not pass_config.include_domains_relaxed:
                    chan_set = set(phase.channels or []) if phase else set()
                    if chan_set and chan_set.issubset({EvidenceChannel.AUTHORITATIVE, EvidenceChannel.REPUTABLE_NEWS}):
                        include_domains = get_trusted_domains_by_lang(phase.locale or "en") if phase else None
                else:
                    domains_relaxed = True
                
                # Execute search
                if hasattr(self.search_mgr, "search_phase"):
                    raw_result = await self.search_mgr.search_phase(
                        query_text,
                        topic=pass_config.topic or topic,
                        depth=pass_config.search_depth,
                        max_results=pass_config.max_results,
                        include_domains=include_domains,
                    )
                else:
                    raw_result = await self.search_mgr.search_unified(
                        query=query_text,
                        topic=pass_config.topic or topic,
                        intent=pass_config.search_depth,
                    )
                
                tavily_calls += 1
                
                # Parse result
                sources: list[dict]
                if isinstance(raw_result, tuple) and len(raw_result) == 2:
                    _, sources = raw_result
                elif isinstance(raw_result, list):
                    sources = raw_result
                else:
                    sources = []
                
                sources = canonicalize_sources(sources)
                pass_sources.extend(sources)
                all_sources.extend(sources)
                
                # Compute outcome (with config for consistent thresholds)
                outcome = compute_retrieval_outcome(all_sources, config)
                
                # Log escalation event
                trace_escalation_pass(
                    claim_id=claim_id,
                    pass_config=pass_config,
                    query_id=query_id,
                    reason_codes=reason_codes,
                    outcome=outcome,
                    include_domains_count=len(include_domains) if include_domains else None,
                )
                
                # Check early stop
                should_stop, stop_reason = should_stop_escalation(outcome, config)
                if should_stop:
                    trace_search_stop(claim_id, pass_config.pass_id, stop_reason, outcome)
                    trace_search_summary(
                        claim_id=claim_id,
                        passes_executed=passes_executed,
                        tavily_calls=tavily_calls,
                        final_outcome=outcome,
                        domains_relaxed=domains_relaxed,
                    )
                    return all_sources, outcome, passes_executed, tavily_calls, domains_relaxed
            
            # Compute reason codes for next pass (with config for consistent thresholds)
            pass_outcome = compute_retrieval_outcome(pass_sources, config)
            reason_codes = compute_escalation_reason_codes(pass_outcome, config)
        
        # Ladder exhausted
        final_outcome = compute_retrieval_outcome(all_sources, config)
        trace_search_summary(
            claim_id=claim_id,
            passes_executed=passes_executed,
            tavily_calls=tavily_calls,
            final_outcome=final_outcome,
            domains_relaxed=domains_relaxed,
        )
        return all_sources, final_outcome, passes_executed, tavily_calls, domains_relaxed


    async def _search_by_phase(
        self,
        claim: Claim,
        phase: Phase | None,
        query_override: str | None = None,
    ) -> list[dict]:
        """
        T18: Execute search for a phase.
        
        Maps:
        - Phase.channels → domain filters (include/exclude)
        - Phase.search_depth → Tavily depth parameter
        - Phase.locale → Tavily language/region
        - Phase.max_results → k parameter
        """
        # Get query from claim
        query = query_override if query_override is not None else self._select_claim_query(claim)
        if not query:
            return []

        # Topic selection from structured claim fields (not text heuristics)
        topic, topic_reasons = select_topic_from_claim(claim)
        claim_id = claim.get("id", "unknown")
        trace_topic_selected(claim_id, topic, topic_reasons)

        # Map phase search depth to Tavily depth.
        depth = "advanced" if phase and phase.search_depth == "advanced" else "basic"

        # Optional include_domains derived from the phase channels.
        include_domains: list[str] | None = None
        chan_set = set(phase.channels or []) if phase else set()
        if chan_set and chan_set.issubset({EvidenceChannel.AUTHORITATIVE, EvidenceChannel.REPUTABLE_NEWS}):
            # Restrict to vetted domains for high-quality phases.
            # NOTE: This is a registry-based restriction; TLD-based authority (e.g. .gov) is
            # handled by post-filtering below (we cannot express it via include_domains).
            include_domains = get_trusted_domains_by_lang(phase.locale or "en")
        else:
            # Do not restrict for phases that include local_media/social/low_reliability_web.
            include_domains = None

        # Execute search via SearchManager.
        #
        # Prefer the `search_phase()` primitive (respects depth/max_results/domains).
        # Keep compatibility with older mocks by accepting either tuple or list results.
        if hasattr(self.search_mgr, "search_phase"):
            raw_result = await self.search_mgr.search_phase(
                query,
                topic=topic,
                depth=depth,
                max_results=phase.max_results if phase else 3,
                include_domains=include_domains,
            )
        else:
            raw_result = await self.search_mgr.search_unified(
                query=query,
                topic=topic,
                intent=phase.search_depth,
            )

        sources: list[dict]
        if (
            isinstance(raw_result, tuple)
            and len(raw_result) == 2
            and isinstance(raw_result[1], list)
        ):
            _, sources = raw_result
        elif isinstance(raw_result, list):
            sources = raw_result
        else:
            cid = claim.get("id", "unknown") if isinstance(claim, dict) else "unknown"
            logger.warning(
                "[Orchestration] Unexpected search result type for claim %s phase %s: %s",
                cid, phase.phase_id, type(raw_result).__name__
            )
            return []

        normalized = canonicalize_sources(sources)

        # Apply channel filtering (best-effort).
        if phase and phase.channels:
            # Normalize phase.channels to string values (they may be EvidenceChannel enums or strings)
            allowed: set[str] = set()
            for x in (phase.channels or []):
                if hasattr(x, "value"):
                    allowed.add(str(x.value))
                else:
                    raw = str(x)
                    allowed.add(raw.strip().lower().replace("-", "_").replace(" ", "_"))

            filtered: list[dict] = []
            for src in normalized:
                url = src.get("url", "")
                if not url:
                    continue
                domain = extract_domain(url)
                chan = get_domain_tier(domain)  # EvidenceChannel enum
                # Normalize to string key for comparison
                chan_key = chan.value if hasattr(chan, "value") else str(chan)

                passed = chan_key in allowed
                Trace.event("search.channel_filter.item", {
                    "domain": domain,
                    "chan_key": chan_key,
                    "allowed": list(allowed),
                    "passed": passed,
                })

                if passed:
                    filtered.append(src)
            normalized = filtered

        # Limit results to phase.max_results
        if phase and len(normalized) > phase.max_results:
            normalized = normalized[:phase.max_results]

        Trace.event(
            "search.filtered",
            {
                "claim_id": claim.get("id", "unknown"),
                "phase_id": phase.phase_id if phase else None,
                "query": query,
                "raw_count": len(sources),
                "normalized_count": len(normalized),
                "max_results": phase.max_results if phase else None,
            },
        )

        return normalized

    async def run_claim_phases(
        self,
        claim: Claim,
        phases: list[Phase],
        existing_sources: list[dict] | None = None,
    ) -> list[dict]:
        """
        Execute phases for a single claim with early exit.
        
        This is a simpler interface for single-claim execution.
        
        Args:
            claim: The claim to search for
            phases: List of phases to execute
            existing_sources: Already collected sources (optional)
            
        Returns:
            List of all sources collected
        """
        claim_id = claim.get("id", "unknown")
        sources = list(existing_sources) if existing_sources else []

        for phase in phases:
            # T20: Trace phase start
            Trace.event("phase.started", {
                "claim_id": claim_id,
                "phase_id": phase.phase_id,
            })

            try:
                phase_sources = await self._search_by_phase(claim, phase)
                sources.extend(phase_sources)

                # T20: Trace completion
                Trace.event("phase.completed", {
                    "claim_id": claim_id,
                    "phase_id": phase.phase_id,
                    "results_count": len(phase_sources),
                })

                # Check sufficiency
                sufficiency = check_sufficiency_for_claim(claim, sources)

                if sufficiency.status == SufficiencyStatus.SUFFICIENT:
                    # T20: Trace stop
                    remaining = [p.phase_id for p in phases if phases.index(p) > phases.index(phase)]
                    Trace.event("phase.stopped", {
                        "claim_id": claim_id,
                        "phase_id": phase.phase_id,
                        "reason": "sufficiency_met",
                        "rule": sufficiency.rule_matched,
                        "skipped_phases": remaining,
                    })
                    break
                if sufficiency.status == SufficiencyStatus.SKIP:
                    remaining = [p.phase_id for p in phases if phases.index(p) > phases.index(phase)]
                    Trace.event("phase.stopped", {
                        "claim_id": claim_id,
                        "phase_id": phase.phase_id,
                        "reason": "sufficiency_skip",
                        "rule": sufficiency.rule_matched,
                        "skipped_phases": remaining,
                    })
                    break
                else:
                    # T20: Trace continue
                    Trace.event("phase.continue", {
                        "claim_id": claim_id,
                        "phase_id": phase.phase_id,
                        "reason": sufficiency.reason,
                    })

            except Exception as e:
                logger.warning("[Orchestration] Phase %s failed: %s", phase.phase_id, e)
                Trace.event("phase.error", {
                    "claim_id": claim_id,
                    "phase_id": phase.phase_id,
                    "error": str(e),
                })
                # Fail-soft: continue to next phase
                continue

        return sources
