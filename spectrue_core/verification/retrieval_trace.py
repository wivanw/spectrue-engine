# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

"""
Retrieval trace formatter for per-claim analysis.

Converts internal ExecutionState/ClaimExecutionState into the
API-facing RetrievalTrace structure for ClaimFrame.
"""

from __future__ import annotations

from spectrue_core.schema.claim_frame import (
    RetrievalHop as RetrievalHopFrame,
    RetrievalTrace,
)
from spectrue_core.verification.execution_plan import (
    ClaimExecutionState,
    RetrievalHop as ExecutionRetrievalHop,
)


def format_retrieval_hop(hop: ExecutionRetrievalHop) -> RetrievalHopFrame:
    """
    Convert internal RetrievalHop to API-facing format.
    
    Args:
        hop: Internal retrieval hop from execution state
    
    Returns:
        RetrievalHopFrame for API response
    """
    return RetrievalHopFrame(
        hop_index=hop.hop_index,
        query=hop.query,
        decision=hop.decision,
        reason=hop.decision_reason,
        phase_id=None,  # Not tracked in internal hop
        query_type=hop.search_depth,
        results_count=hop.results_count,
        retrieval_eval={"cost_credits": hop.cost_credits} if hop.cost_credits else None,
    )


def format_retrieval_trace(state: ClaimExecutionState) -> RetrievalTrace:
    """
    Convert ClaimExecutionState to API-facing RetrievalTrace.
    
    Args:
        state: Internal execution state for a claim
    
    Returns:
        RetrievalTrace for ClaimFrame
    """
    hops = tuple(format_retrieval_hop(hop) for hop in state.hops)

    return RetrievalTrace(
        phases_completed=tuple(state.phases_completed),
        hops=hops,
        stop_reason=state.stop_reason,
        sufficiency_reason=state.sufficiency_reason if state.is_sufficient else None,
    )


def create_empty_retrieval_trace() -> RetrievalTrace:
    """
    Create an empty retrieval trace for claims with no retrieval.
    
    Used when verification_target is NONE or claim is skipped.
    """
    return RetrievalTrace(
        phases_completed=(),
        hops=(),
        stop_reason="no_retrieval_needed",
        sufficiency_reason=None,
    )


def create_error_retrieval_trace(error_message: str) -> RetrievalTrace:
    """
    Create a retrieval trace indicating an error occurred.
    
    Args:
        error_message: Error description
    
    Returns:
        RetrievalTrace with error information
    """
    return RetrievalTrace(
        phases_completed=(),
        hops=(),
        stop_reason=f"error: {error_message}",
        sufficiency_reason=None,
    )
