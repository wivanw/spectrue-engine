from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any

from spectrue_core.use_cases.verification.orchestration.execution_state import ClaimExecutionState

@dataclass(frozen=True)
class ClaimExecutionContext:
    """
    Immutable execution context for a single claim in Deep Mode.
    Ensures that state and evidence mutation for one claim cannot bleed into another.
    """
    claim_id: str
    claim: dict[str, Any]
    retrieval_plan: dict[str, Any] | None = None
    evidence_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    state: ClaimExecutionState = field(default_factory=lambda: ClaimExecutionState(claim_id="unknown"))

    @classmethod
    def create(
        cls,
        claim: dict[str, Any],
        retrieval_plan: dict[str, Any] | None = None,
        evidence_items: list[dict[str, Any]] | None = None,
        state: ClaimExecutionState | None = None,
    ) -> ClaimExecutionContext:
        """
        Factory to create an isolated, immutable context.
        Provides deep copies of mutable components to prevent cross-claim bleeding.
        """
        claim_id = str(claim.get("id") or claim.get("claim_id") or "unknown")
        
        # Deep copy to ensure no shared mutable state
        safe_claim = copy.deepcopy(claim)
        safe_plan = copy.deepcopy(retrieval_plan) if retrieval_plan else None
        
        # Convert evidence list to immutable tuple of deep-copied dicts
        safe_evidence = tuple(copy.deepcopy(ev) for ev in (evidence_items or []))
        
        # State isolation
        if state is None:
            safe_state = ClaimExecutionState(claim_id=claim_id)
        else:
            safe_state = copy.deepcopy(state)
            safe_state.claim_id = claim_id
            
        return cls(
            claim_id=claim_id,
            claim=safe_claim,
            retrieval_plan=safe_plan,
            evidence_items=safe_evidence,
            state=safe_state,
        )

    def with_evidence(self, new_evidence: list[dict[str, Any]]) -> ClaimExecutionContext:
        """Return a new context with added evidence items."""
        merged_evidence = list(self.evidence_items) + new_evidence
        return ClaimExecutionContext.create(
            claim=self.claim,
            retrieval_plan=self.retrieval_plan,
            evidence_items=merged_evidence,
            state=self.state,
        )

    def with_state_update(self, state: ClaimExecutionState) -> ClaimExecutionContext:
        """Return a new context with an updated state."""
        return ClaimExecutionContext.create(
            claim=self.claim,
            retrieval_plan=self.retrieval_plan,
            evidence_items=list(self.evidence_items),
            state=state,
        )

    def with_retrieval_plan(self, plan: dict[str, Any]) -> ClaimExecutionContext:
        """Return a new context with an updated retrieval plan."""
        return ClaimExecutionContext.create(
            claim=self.claim,
            retrieval_plan=plan,
            evidence_items=list(self.evidence_items),
            state=self.state,
        )
