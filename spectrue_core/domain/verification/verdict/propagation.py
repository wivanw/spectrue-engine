"""Belief Propagation through Claim Context Graph domain logic."""

from __future__ import annotations

from typing import List, Any
from .model import BeliefState, ScoringTraceStep


def propagate_belief(graph: Any) -> List[ScoringTraceStep]:
    """
    Propagates belief through the claim graph using message passing on DAG.
    Updates `propagated_belief` on each node.
    """
    trace = []
    sorted_ids = graph.topological_sort()

    for idx, claim_id in enumerate(sorted_ids):
        node = graph.get_node(claim_id)
        if not node:
            continue

        current_belief = node.local_belief or BeliefState(log_odds=0.0)
        incoming_edges = graph.get_incoming_edges(claim_id)

        total_message_log_odds = 0.0

        for edge in incoming_edges:
            source_node = graph.get_node(edge.source_id)
            if not source_node or not source_node.propagated_belief:
                continue

            source_log_odds = source_node.propagated_belief.log_odds

            # CONTRADICTS: If source is TRUE, target is more likely FALSE
            # SUPPORTS/ENTAILS: If source is TRUE, target is more likely TRUE
            sign = 1.0
            rel = str(getattr(edge, "relation", "")).lower()
            if "contradict" in rel:
                sign = -1.0

            message = source_log_odds * edge.weight * sign
            total_message_log_odds += message

        final_log_odds = current_belief.log_odds + total_message_log_odds

        node.propagated_belief = BeliefState(
            log_odds=final_log_odds, 
            confidence=current_belief.confidence
        )
        
        # Add trace entry using ScoringTraceStep
        description = f"Propagated belief for {claim_id}"
        if incoming_edges:
            rels = [str(getattr(e, "relation", "")).upper() for e in incoming_edges]
            description += f" via {', '.join(rels)}"
        
        trace.append(ScoringTraceStep(
            step_id=idx,
            description=description,
            delta=total_message_log_odds,
            new_belief=final_log_odds,
        ))

    return trace


def propagation_routing_signals(graph: Any) -> dict[str, float]:
    """
    Extract propagation outputs as routing-friendly signals.
    """
    signals: dict[str, float] = {}
    for node_id in graph.topological_sort():
        node = graph.get_node(node_id)
        if not node or not node.propagated_belief:
            continue
        signals[node_id] = float(node.propagated_belief.log_odds)
    return signals
