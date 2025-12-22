# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M80: Phase Runner

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
from typing import Any, TYPE_CHECKING

from spectrue_core.verification.execution_plan import (
    Phase,
    ExecutionPlan,
    ClaimExecutionState,
    ExecutionState,
)
from spectrue_core.verification.sufficiency import (
    check_sufficiency_for_claim,
    SufficiencyStatus,
    get_domain_tier,
)
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.trusted_sources import get_trusted_domains_by_lang
from spectrue_core.schema.claim_metadata import EvidenceChannel
from spectrue_core.verification.source_utils import canonicalize_sources, extract_domain

if TYPE_CHECKING:
    from spectrue_core.verification.search_mgr import SearchManager
    from spectrue_core.verification.evidence_pack import Claim

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
    ) -> None:
        """
        Initialize PhaseRunner.
        
        Args:
            search_mgr: SearchManager for executing searches
            max_concurrent: Maximum concurrent searches per phase
            progress_callback: Optional async callback for progress updates
        """
        self.search_mgr = search_mgr
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.execution_state = ExecutionState()
        self.progress_callback = progress_callback
    
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
        
        # Determine phase order (collect all unique phases in order)
        phase_order = ["A-light", "A", "A-origin", "B", "C", "D"]
        
        for phase_id in phase_order:
            # Get claims that need this phase (and aren't already sufficient)
            claims_for_phase = self._get_claims_needing_phase(
                claims=claims,
                execution_plan=execution_plan,
                phase_id=phase_id,
            )
            
            if not claims_for_phase:
                continue
            
            logger.info(
                "[M80] Running Phase %s for %d claim(s)",
                phase_id, len(claims_for_phase)
            )
            
            # Emit progress: searching_phase_X
            if self.progress_callback:
                status_key = f"searching_phase_{phase_id.lower().replace('-', '_')}"
                try:
                    await self.progress_callback(status_key)
                except Exception as e:
                    logger.warning("[M80] Progress callback failed: %s", e)
            
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
                        
                        logger.info(
                            "[M80] Claim %s sufficient after Phase %s (rule=%s), skipping %s",
                            claim_id, result.phase_id, sufficiency.rule_matched, remaining
                        )
                    else:
                        # T20: Trace continue
                        Trace.event("phase.continue", {
                            "claim_id": claim_id,
                            "phase_id": result.phase_id,
                            "reason": sufficiency.reason,
                        })
        
        return evidence
    
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
                    "[M80] Phase %s failed for claim %s: %s",
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
    
    async def _search_by_phase(
        self,
        claim: Claim,
        phase: Phase,
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
        queries = claim.get("search_queries", [])
        if not queries:
            # Try query_candidates
            candidates = claim.get("query_candidates", [])
            if candidates:
                queries = [c.get("text") for c in candidates if c.get("text")]
        
        if not queries:
            # Fallback to normalized text
            queries = [claim.get("normalized_text", "") or claim.get("text", "")]
        
        # Use first query (or could combine)
        query = queries[0] if queries else ""
        if not query:
            return []
        
        # Determine topic (news vs general) from claim's search_method when available.
        # Fall back to a conservative default mapping for backward compatibility.
        topic = "general"
        claim_method = claim.get("search_method", "")
        if claim_method in ("news", "general_search"):
            topic = "news" if claim_method == "news" else "general"
        else:
            topic = "general" if phase.search_depth == "advanced" else "news"

        # Map phase search depth to Tavily depth.
        depth = "advanced" if phase.search_depth == "advanced" else "basic"

        # Optional include_domains derived from the phase channels.
        include_domains: list[str] | None = None
        chan_set = set(phase.channels or [])
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
        # Prefer the M83 `search_phase()` primitive (respects depth/max_results/domains).
        # Keep compatibility with older mocks by accepting either tuple or list results.
        if hasattr(self.search_mgr, "search_phase"):
            raw_result = await self.search_mgr.search_phase(
                query,
                topic=topic,
                depth=depth,
                max_results=phase.max_results,
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
            logger.warning(
                "[M80] Unexpected search result type for claim %s phase %s: %s",
                claim_id, phase.phase_id, type(raw_result).__name__
            )
            return []
        
        normalized = canonicalize_sources(sources)

        # Apply channel filtering (best-effort).
        if phase.channels:
            allowed = set(phase.channels)
            filtered: list[dict] = []
            for src in normalized:
                url = src.get("url", "")
                if not url:
                    continue
                domain = extract_domain(url)
                tier = get_domain_tier(domain)
                if tier in allowed:
                    filtered.append(src)
            normalized = filtered
        
        # Limit results to phase.max_results
        if len(normalized) > phase.max_results:
            normalized = normalized[:phase.max_results]
        
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
                else:
                    # T20: Trace continue
                    Trace.event("phase.continue", {
                        "claim_id": claim_id,
                        "phase_id": phase.phase_id,
                        "reason": sufficiency.reason,
                    })
                    
            except Exception as e:
                logger.warning("[M80] Phase %s failed: %s", phase.phase_id, e)
                Trace.event("phase.error", {
                    "claim_id": claim_id,
                    "phase_id": phase.phase_id,
                    "error": str(e),
                })
                # Fail-soft: continue to next phase
                continue
        
        return sources
