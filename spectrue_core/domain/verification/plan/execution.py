"""
Execution Plan logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import BudgetClass, Phase


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for all claims.
    
    Maps each claim_id to its list of phases to execute.
    Phases are executed in order; early exit on sufficiency.
    """
    claim_phases: dict[str, list[Phase]] = field(default_factory=dict)
    """Mapping of claim_id to list of phases."""

    budget_class: BudgetClass = BudgetClass.BALANCED
    """Budget classification used to build this plan."""

    # Pipeline profile metadata
    profile_name: str | None = None
    """Name of the pipeline profile used to build this plan (M113)."""

    profile_version: str | None = None
    """Version of the pipeline profile (M113)."""

    overrides: dict[str, Any] | None = None
    """Per-run overrides applied to the profile (M113)."""

    max_credits: int | None = None
    """Maximum credits for this run (M113)."""

    def summary(self) -> str:
        """Get human-readable summary of the plan."""
        lines = [f"ExecutionPlan(budget={self.budget_class.value}, total_claims={len(self.claim_phases)})"]
        for cid, phases in self.claim_phases.items():
            p_ids = [p.phase_id for p in phases]
            lines.append(f"  {cid}: {', '.join(p_ids)}")
        return "\n".join(lines)

    def get_phases(self, claim_id: str) -> list[Phase]:
        """Get phases for a claim. Returns empty list if not found."""
        return self.claim_phases.get(claim_id, [])

    def add_claim(self, claim_id: str, phases: list[Phase]) -> None:
        """Add phases for a claim."""
        self.claim_phases[claim_id] = phases

    def get_all_phase_ids(self) -> set[str]:
        """Get all unique phase IDs in the plan."""
        phase_ids: set[str] = set()
        for phases in self.claim_phases.values():
            for phase in phases:
                phase_ids.add(phase.phase_id)
        return phase_ids

    def get_claims_needing_phase(self, phase_id: str) -> list[str]:
        """Get list of claim IDs that need a specific phase."""
        return [
            claim_id
            for claim_id, phases in self.claim_phases.items()
            if any(p.phase_id == phase_id for p in phases)
        ]

    @property
    def total_phases(self) -> int:
        """Total number of phase executions across all claims."""
        return sum(len(phases) for phases in self.claim_phases.values())

    @property
    def max_depth(self) -> int:
        """Maximum number of phases for any single claim."""
        if not self.claim_phases:
            return 0
        return max(len(phases) for phases in self.claim_phases.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan for tracing/storage."""
        return {
            "claim_phases": {
                cid: [p.to_dict() for p in phases]
                for cid, phases in self.claim_phases.items()
            },
            "budget_class": self.budget_class.value,
            "stats": {
                "total_claims": len(self.claim_phases),
                "total_phases": self.total_phases,
                "max_depth": self.max_depth,
            },
            "profile": {
                "name": self.profile_name,
                "version": self.profile_version,
                "max_credits": self.max_credits,
            }
        }
