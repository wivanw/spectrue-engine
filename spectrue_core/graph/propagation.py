"""
M104: Belief Propagation through Claim Context Graph

Mathematical Foundation:
------------------------
This module implements a simplified Belief Propagation (BP) algorithm for
acyclic claim graphs (DAGs). The algorithm propagates probabilistic beliefs
through semantic relationships between claims.

Key concepts:

1. **Message Passing on DAGs**:
   - In a DAG, topological sorting guarantees that when we process a node,
     all its ancestors have already been processed.
   - This allows single-pass belief propagation without iterative convergence.

2. **Log-Odds Additivity**:
   - Beliefs are represented in log-odds space where updates are additive.
   - Message: m(A→B) = log_odds(A) × weight(A,B) × sign(relation)
   - Target belief: log_odds(B) = local(B) + Σ m(parent → B)

3. **Semantic Relationships**:
   - SUPPORTS: Positive influence (sign = +1)
     If A is true and A supports B, B is more likely true.
   - CONTRADICTS: Negative influence (sign = -1)
     If A is true and A contradicts B, B is more likely false.
   - Edge weight: Semantic strength of the relationship [0, 1]

4. **Neutral Node Property**:
   - A node with log_odds=0.0 (probability 0.5) exerts no influence.
   - This is desirable: uncertain claims shouldn't affect descendants.

References:
- Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems.
- Koller, D. & Friedman, N. (2009). Probabilistic Graphical Models.
"""

from typing import List
from spectrue_core.graph.context import ClaimContextGraph
from spectrue_core.schema.scoring import BeliefState, ScoringTraceStep, RelationType


def propagate_belief(graph: ClaimContextGraph) -> List[ScoringTraceStep]:
    """
    Propagates belief through the claim graph using message passing on DAG.
    Updates `propagated_belief` on each node.
    Returns a trace of updates for explainability (FR-009).
    
    Algorithm (Single-Pass BP for DAGs):
    ------------------------------------
    1. Topologically sort nodes (ensures dependencies processed first)
    2. For each node in order:
       a. Start with local belief (from direct evidence)
       b. Sum incoming messages from all parent nodes
       c. Message = parent.belief × edge.weight × sign(relation)
       d. Final belief = local + sum(messages)
    
    Complexity: O(V + E) where V=nodes, E=edges
    """
    trace = []
    step_id = 0
    
    # Topological sort ensures DAG property: parents processed before children
    # If graph has cycles (circular reasoning), fallback to arbitrary order
    sorted_ids = graph.topological_sort()
    
    for claim_id in sorted_ids:
        node = graph.get_node(claim_id)
        if not node:
            continue
            
        # Local belief: Evidence directly about THIS claim
        # Neutral (log_odds=0.0) means P(true)=0.5 (no direct evidence)
        current_belief = node.local_belief or BeliefState(log_odds=0.0)
        
        # Collect messages from parent nodes (claims that influence this one)
        incoming_edges = graph.get_incoming_edges(claim_id)
        
        total_message_log_odds = 0.0
        
        for edge in incoming_edges:
            source_node = graph.get_node(edge.source_id)
            
            # Skip if source hasn't been processed (shouldn't happen in topo sort)
            # or has no propagated belief yet
            if not source_node or not source_node.propagated_belief:
                continue
                
            source_log_odds = source_node.propagated_belief.log_odds
            
            # Determine influence direction based on semantic relationship
            # CONTRADICTS: If source is TRUE, target is more likely FALSE
            # SUPPORTS/ENTAILS: If source is TRUE, target is more likely TRUE
            sign = 1.0
            if edge.relation == RelationType.CONTRADICTS:
                sign = -1.0
                
            # Message formula:
            # m = source_belief × edge_weight × sign
            #
            # Properties:
            # - Strong source belief (high |log_odds|) → strong influence
            # - Neutral source (log_odds≈0) → no influence (desirable!)
            # - High weight → strong coupling
            # - CONTRADICTS → inverts the influence
            message = source_log_odds * edge.weight * sign
            
            total_message_log_odds += message
            
            # Record for explainability trace
            step_id += 1
            trace.append(ScoringTraceStep(
                step_id=step_id,
                description=f"Propagation from {edge.source_id} to {claim_id} ({edge.relation})",
                delta=message,
                new_belief=current_belief.log_odds + total_message_log_odds
            ))
            
        # Final belief = Local evidence + Sum of incoming messages
        # This is the core BP update rule for log-odds representation
        final_log_odds = current_belief.log_odds + total_message_log_odds
        
        node.propagated_belief = BeliefState(
            log_odds=final_log_odds, 
            confidence=current_belief.confidence
        )
        
    return trace


def propagation_routing_signals(graph: ClaimContextGraph) -> dict[str, float]:
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
