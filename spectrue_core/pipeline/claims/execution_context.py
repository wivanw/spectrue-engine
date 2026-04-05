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
    search_queries: tuple[str, ...] = field(default_factory=tuple)
    entities: tuple[str, ...] = field(default_factory=tuple)

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
            
        # Extract structured fields from claim
        safe_queries = tuple(str(q) for q in (claim.get("search_queries") or []) if isinstance(q, str))
        safe_entities = tuple(sorted(set(
            str(e) for e in (claim.get("entities") or []) if isinstance(e, str)
        )))

        return cls(
            claim_id=claim_id,
            claim=safe_claim,
            retrieval_plan=safe_plan,
            evidence_items=safe_evidence,
            state=safe_state,
            search_queries=safe_queries,
            entities=safe_entities,
        )

    def with_evidence(self, new_evidence: list[dict[str, Any]]) -> ClaimExecutionContext:
        """Return a new context with added evidence items."""
        merged_evidence = list(self.evidence_items) + new_evidence
        ctx = ClaimExecutionContext.create(
            claim=self.claim,
            retrieval_plan=self.retrieval_plan,
            evidence_items=merged_evidence,
            state=self.state,
        )
        # Preserve search_queries/entities from original context
        return ClaimExecutionContext(
            claim_id=ctx.claim_id,
            claim=ctx.claim,
            retrieval_plan=ctx.retrieval_plan,
            evidence_items=ctx.evidence_items,
            state=ctx.state,
            search_queries=self.search_queries,
            entities=self.entities,
        )

    def with_state_update(self, state: ClaimExecutionState) -> ClaimExecutionContext:
        """Return a new context with an updated state."""
        ctx = ClaimExecutionContext.create(
            claim=self.claim,
            retrieval_plan=self.retrieval_plan,
            evidence_items=list(self.evidence_items),
            state=state,
        )
        return ClaimExecutionContext(
            claim_id=ctx.claim_id,
            claim=ctx.claim,
            retrieval_plan=ctx.retrieval_plan,
            evidence_items=ctx.evidence_items,
            state=ctx.state,
            search_queries=self.search_queries,
            entities=self.entities,
        )

    def with_retrieval_plan(self, plan: dict[str, Any]) -> ClaimExecutionContext:
        """Return a new context with an updated retrieval plan."""
        ctx = ClaimExecutionContext.create(
            claim=self.claim,
            retrieval_plan=plan,
            evidence_items=list(self.evidence_items),
            state=self.state,
        )
        return ClaimExecutionContext(
            claim_id=ctx.claim_id,
            claim=ctx.claim,
            retrieval_plan=ctx.retrieval_plan,
            evidence_items=ctx.evidence_items,
            state=ctx.state,
            search_queries=self.search_queries,
            entities=self.entities,
        )
