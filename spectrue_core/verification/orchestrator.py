# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
M80: Claim Orchestrator

Builds ExecutionPlan from claims + metadata.
Phase selection is driven by:
- verification_target: Controls whether to search at all
- claim_role: Affects priority in queue
- metadata_confidence: Triggers fail-open (A-light injection)
- search_locale_plan: Determines locales for phases
- retrieval_policy: Filters allowed channels

Waterfall Phases:
- A: Primary locale, authoritative + reputable, basic, k=3
- B: Primary locale, + local media, advanced, k=5
- C: Fallback locale, authoritative + reputable, basic, k=3
- D: All channels, advanced, k=7

Budget Classes:
- MINIMAL: Phase A only
- STANDARD: Phases A + B
- DEEP: All phases (A/B/C/D)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    VerificationTarget,
    MetadataConfidence,
    RetrievalPolicy,
)
from spectrue_core.verification.execution_plan import (
    Phase,
    ExecutionPlan,
    BudgetClass,
    phase_a,
    phase_a_light,
    phase_a_origin,
    phase_b,
    phase_c,
    phase_d,
)
from spectrue_core.utils.trace import Trace

if TYPE_CHECKING:
    from spectrue_core.verification.evidence_pack import Claim

logger = logging.getLogger(__name__)


class ClaimOrchestrator:
    """
    Builds ExecutionPlan from a list of claims.
    
    Uses claim metadata to determine:
    - Which phases each claim needs
    - Whether to inject fail-open phases
    - How to handle different verification targets
    
    Example:
        orchestrator = ClaimOrchestrator()
        plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)
    """
    
    def __init__(self) -> None:
        """Initialize the orchestrator."""
        pass
    
    def build_execution_plan(
        self,
        claims: list[Claim],
        budget_class: BudgetClass = BudgetClass.STANDARD,
    ) -> ExecutionPlan:
        """
        Build an ExecutionPlan from claims.
        
        Each claim gets a list of phases based on its metadata.
        Claims with verification_target=NONE get minimal/no phases.
        Claims with low metadata_confidence get A-light prepended.
        
        Args:
            claims: List of claims with metadata
            budget_class: Budget for search depth
            
        Returns:
            ExecutionPlan with claim_phases mapping
        """
        plan = ExecutionPlan(budget_class=budget_class)
        
        stats = {
            "total": len(claims),
            "reality": 0,
            "attribution": 0,
            "existence": 0,
            "none": 0,
            "fail_open": 0,
        }
        
        for claim in claims:
            claim_id = claim.get("id", "unknown")
            metadata = claim.get("metadata")
            
            if metadata is None:
                # No metadata: treat as core reality claim with defaults
                phases = self._generate_phases_for_reality(
                    claim_id=claim_id,
                    locale="en",
                    fallback_locale="en",
                    budget_class=budget_class,
                )
                stats["reality"] += 1
            else:
                phases = self._generate_phases(
                    claim_id=claim_id,
                    metadata=metadata,
                    budget_class=budget_class,
                )
                
                # Update stats
                target = metadata.verification_target
                if target == VerificationTarget.REALITY:
                    stats["reality"] += 1
                elif target == VerificationTarget.ATTRIBUTION:
                    stats["attribution"] += 1
                elif target == VerificationTarget.EXISTENCE:
                    stats["existence"] += 1
                else:
                    stats["none"] += 1
                
                # Check if fail-open was applied
                if phases and phases[0].phase_id == "A-light" and metadata.metadata_confidence == MetadataConfidence.LOW:
                    stats["fail_open"] += 1
            
            plan.add_claim(claim_id, phases)
        
        Trace.event("orchestrator.plan_built", {
            "budget_class": budget_class.value,
            "claims_count": len(claims),
            "stats": stats,
            "plan_summary": plan.summary(),
        })
        
        logger.debug(
            "[M80] Orchestrator: %d claims, budget=%s | reality=%d, attribution=%d, existence=%d, none=%d, fail_open=%d",
            len(claims), budget_class.value, stats["reality"], stats["attribution"], 
            stats["existence"], stats["none"], stats["fail_open"]
        )
        
        return plan

    def build_plan(
        self,
        claims: list[Claim],
        budget_class: str | BudgetClass = BudgetClass.STANDARD,
    ) -> ExecutionPlan:
        """Alias for build_execution_plan to match pipeline call."""
        if isinstance(budget_class, str):
            try:
                budget_class = BudgetClass(budget_class)
            except ValueError:
                budget_class = BudgetClass.STANDARD
                
        return self.build_execution_plan(claims, budget_class)
    
    def _generate_phases(
        self,
        claim_id: str,
        metadata: ClaimMetadata,
        budget_class: BudgetClass,
    ) -> list[Phase]:
        """
        Generate phases for a claim based on its metadata.
        
        Routing logic by verification_target:
        - NONE: Empty phases (no search) or A-light if fail-open
        - REALITY: Full phases A→D based on budget
        - ATTRIBUTION: Origin-focused phases
        - EXISTENCE: Similar to reality but lighter
        """
        target = metadata.verification_target
        confidence = metadata.metadata_confidence
        locale_plan = metadata.search_locale_plan
        
        # Get locales
        primary_locale = locale_plan.primary if locale_plan else "en"
        fallback_locale = (locale_plan.fallback[0] if locale_plan and locale_plan.fallback else "en")
        
        phases: list[Phase] = []
        
        # 1. Handle NONE target (predictions, opinions, horoscopes)
        if target == VerificationTarget.NONE:
            if confidence == MetadataConfidence.LOW:
                # Fail-open: still do minimal search
                phases = [phase_a_light(primary_locale)]
                logger.debug("[M80] Claim %s: target=none, confidence=low → fail-open A-light", claim_id)
            else:
                # Skip search entirely
                phases = []
                logger.debug("[M80] Claim %s: target=none → skip search", claim_id)
            return phases
        
        # 2. Handle ATTRIBUTION target
        if target == VerificationTarget.ATTRIBUTION:
            phases = self._generate_phases_for_attribution(
                claim_id=claim_id,
                locale=primary_locale,
                fallback_locale=fallback_locale,
                budget_class=budget_class,
                confidence=confidence,
            )
            return self._apply_retrieval_policy(phases, metadata.retrieval_policy)
        
        # 3. Handle EXISTENCE target
        if target == VerificationTarget.EXISTENCE:
            phases = self._generate_phases_for_existence(
                claim_id=claim_id,
                locale=primary_locale,
                fallback_locale=fallback_locale,
                budget_class=budget_class,
                confidence=confidence,
            )
            return self._apply_retrieval_policy(phases, metadata.retrieval_policy)
        
        # 4. Handle REALITY target (default path)
        phases = self._generate_phases_for_reality(
            claim_id=claim_id,
            locale=primary_locale,
            fallback_locale=fallback_locale,
            budget_class=budget_class,
            confidence=confidence,
        )
        
        # Apply retrieval_policy constraints to the phase list.
        phases = self._apply_retrieval_policy(phases, metadata.retrieval_policy)
        return phases
    
    def _generate_phases_for_reality(
        self,
        claim_id: str,
        locale: str,
        fallback_locale: str,
        budget_class: BudgetClass,
        confidence: MetadataConfidence = MetadataConfidence.HIGH,
    ) -> list[Phase]:
        """
        Generate phases for a reality-checking claim.
        
        Full waterfall: A → B → C → D (based on budget).
        """
        phases: list[Phase] = []
        
        # T12: Fail-open injection for low confidence
        if confidence == MetadataConfidence.LOW:
            phases.append(phase_a_light(locale))
            logger.debug("[M80] Claim %s: low confidence → prepending A-light", claim_id)
        
        # Budget-based phase selection
        if budget_class == BudgetClass.MINIMAL:
            # Only Phase A
            phases.append(phase_a(locale))
        
        elif budget_class == BudgetClass.STANDARD:
            # Phases A + B
            phases.append(phase_a(locale))
            phases.append(phase_b(locale))
        
        elif budget_class == BudgetClass.DEEP:
            # All phases: A → B → C → D
            phases.append(phase_a(locale))
            phases.append(phase_b(locale))
            if fallback_locale != locale:
                phases.append(phase_c(fallback_locale))
            phases.append(phase_d(locale))
        
        return phases
    
    def _generate_phases_for_attribution(
        self,
        claim_id: str,
        locale: str,
        fallback_locale: str,
        budget_class: BudgetClass,
        confidence: MetadataConfidence = MetadataConfidence.HIGH,
    ) -> list[Phase]:
        """
        Generate phases for attribution claims ("X said Y").
        
        Uses origin-focused phases to find the original source.
        """
        phases: list[Phase] = []
        
        # Fail-open injection
        if confidence == MetadataConfidence.LOW:
            phases.append(phase_a_light(locale))
            logger.debug("[M80] Claim %s (attribution): low confidence → prepending A-light", claim_id)
        
        # Origin-focused primary phase
        phases.append(phase_a_origin(locale))
        
        # Standard expansion based on budget
        if budget_class in (BudgetClass.STANDARD, BudgetClass.DEEP):
            phases.append(phase_b(locale))
        
        if budget_class == BudgetClass.DEEP:
            if fallback_locale != locale:
                phases.append(phase_c(fallback_locale))
            phases.append(phase_d(locale))
        
        return phases
    
    def _generate_phases_for_existence(
        self,
        claim_id: str,
        locale: str,
        fallback_locale: str,
        budget_class: BudgetClass,
        confidence: MetadataConfidence = MetadataConfidence.HIGH,
    ) -> list[Phase]:
        """
        Generate phases for existence claims (verify a source/document exists).
        
        Similar to reality but generally needs fewer sources.
        """
        phases: list[Phase] = []
        
        # Fail-open injection
        if confidence == MetadataConfidence.LOW:
            phases.append(phase_a_light(locale))
            logger.debug("[M80] Claim %s (existence): low confidence → prepending A-light", claim_id)
        
        # Existence claims typically need fewer phases
        phases.append(phase_a(locale))
        
        # Only expand on higher budgets
        if budget_class == BudgetClass.DEEP:
            phases.append(phase_b(locale))
            if fallback_locale != locale:
                phases.append(phase_c(fallback_locale))
        
        return phases

    def _apply_retrieval_policy(
        self,
        phases: list[Phase],
        retrieval_policy: RetrievalPolicy,
    ) -> list[Phase]:
        """
        M83: Apply RetrievalPolicy to a list of phases.
        
        - Intersects each Phase.channels with RetrievalPolicy.channels_allowed
        - Drops phases that end up with no channels
        - Sets Phase.use_policy_by_channel for downstream logic (lead_only vs support_ok)
        """
        if not phases:
            return phases

        allowed = set(retrieval_policy.channels_allowed or [])
        out: list[Phase] = []
        for p in phases:
            if allowed:
                p.channels = [c for c in p.channels if c in allowed]
            if not p.channels:
                continue
            p.use_policy_by_channel = {
                c.value: retrieval_policy.get_use_policy(c)
                for c in p.channels
            }
            out.append(p)
        return out
