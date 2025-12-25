"""
T026: Integration test for contradictory claim propagation.

Tests the belief propagation through the ClaimContextGraph when claims
have contradictory relationships.

Scenario: A thesis is supported by one claim but contradicted by another.
The final belief should reflect the combined influence.
"""
from spectrue_core.graph.context import ClaimContextGraph
from spectrue_core.graph.propagation import propagate_belief
from spectrue_core.schema.scoring import (
    BeliefState,
    ClaimEdge,
    ClaimNode,
    ClaimRole,
    RelationType,
)


def test_contradiction_lowers_thesis_belief():
    """
    US3 Validation: Link a true claim to a false thesis via CONTRADICTS edge.
    Verify that the thesis credibility drops.
    
    Setup:
    - Claim A (Evidence): Strong positive belief (log_odds=2.0 ~88%)
    - Thesis B: Neutral initial (log_odds=0.0 ~50%)
    - Edge: A --CONTRADICTS--> B (weight=0.8)
    
    Expected: Thesis B's propagated belief should be negative (< 0).
    """
    graph = ClaimContextGraph()
    
    # Claim A: Strong evidence (true claim)
    claim_a = ClaimNode(
        claim_id="a",
        text="The vaccine is safe and effective",
        role=ClaimRole.SUPPORT,
        local_belief=BeliefState(log_odds=2.0, confidence=0.9),
    )
    
    # Thesis B: Initially neutral
    thesis_b = ClaimNode(
        claim_id="b",
        text="Vaccines are dangerous",
        role=ClaimRole.THESIS,
        local_belief=BeliefState(log_odds=0.0, confidence=0.5),
    )
    
    graph.add_node(claim_a)
    graph.add_node(thesis_b)
    
    # A contradicts B (A's truth makes B less credible)
    edge_ab = ClaimEdge(
        source_id="a",
        target_id="b",
        relation=RelationType.CONTRADICTS,
        weight=0.8,
    )
    graph.add_edge(edge_ab)
    
    # Run propagation
    trace = propagate_belief(graph)
    
    # Verify thesis belief dropped
    thesis_node = graph.get_node("b")
    assert thesis_node is not None
    assert thesis_node.propagated_belief is not None
    
    # Expected: 0.0 + (2.0 * 0.8 * -1) = -1.6
    assert thesis_node.propagated_belief.log_odds < 0.0
    assert thesis_node.propagated_belief.probability < 0.5
    
    # Verify trace recorded the step
    assert len(trace) >= 1
    assert any("CONTRADICTS" in step.description or "contradicts" in step.description for step in trace)


def test_support_raises_thesis_belief():
    """
    Baseline test: A supporting claim should raise the thesis belief.
    
    Setup:
    - Claim A (Evidence): Positive belief (log_odds=1.5)
    - Thesis B: Neutral initial (log_odds=0.0)
    - Edge: A --SUPPORTS--> B (weight=0.7)
    
    Expected: Thesis B's belief should increase.
    """
    graph = ClaimContextGraph()
    
    claim_a = ClaimNode(
        claim_id="a",
        text="Studies show the drug reduces symptoms",
        role=ClaimRole.SUPPORT,
        local_belief=BeliefState(log_odds=1.5, confidence=0.8),
    )
    
    thesis_b = ClaimNode(
        claim_id="b",
        text="The drug is effective",
        role=ClaimRole.THESIS,
        local_belief=BeliefState(log_odds=0.0, confidence=0.5),
    )
    
    graph.add_node(claim_a)
    graph.add_node(thesis_b)
    
    edge_ab = ClaimEdge(
        source_id="a",
        target_id="b",
        relation=RelationType.SUPPORTS,
        weight=0.7,
    )
    graph.add_edge(edge_ab)
    
    propagate_belief(graph)
    
    thesis_node = graph.get_node("b")
    assert thesis_node is not None
    assert thesis_node.propagated_belief is not None
    
    # Expected: 0.0 + (1.5 * 0.7 * 1) = 1.05
    assert thesis_node.propagated_belief.log_odds > 0.0
    assert thesis_node.propagated_belief.probability > 0.5


def test_conflicting_evidence_net_effect():
    """
    Complex scenario: Thesis has both supporting and contradicting evidence.
    The net effect should reflect the combined influence.
    
    Setup:
    - Claim A (Support): log_odds=1.0
    - Claim B (Contradiction): log_odds=2.0 (stronger)
    - Thesis C: Neutral
    - Edge: A --SUPPORTS--> C (weight=0.6)
    - Edge: B --CONTRADICTS--> C (weight=0.8)
    
    Expected: Net effect is negative because B is stronger and contradicts.
    """
    graph = ClaimContextGraph()
    
    support_claim = ClaimNode(
        claim_id="support",
        text="Some evidence for the thesis",
        role=ClaimRole.SUPPORT,
        local_belief=BeliefState(log_odds=1.0, confidence=0.7),
    )
    
    contra_claim = ClaimNode(
        claim_id="contra",
        text="Strong evidence against the thesis",
        role=ClaimRole.COUNTER,
        local_belief=BeliefState(log_odds=2.0, confidence=0.9),
    )
    
    thesis = ClaimNode(
        claim_id="thesis",
        text="The main claim",
        role=ClaimRole.THESIS,
        local_belief=BeliefState(log_odds=0.0, confidence=0.5),
    )
    
    graph.add_node(support_claim)
    graph.add_node(contra_claim)
    graph.add_node(thesis)
    
    # Support edge
    graph.add_edge(ClaimEdge(
        source_id="support",
        target_id="thesis",
        relation=RelationType.SUPPORTS,
        weight=0.6,
    ))
    
    # Contradiction edge
    graph.add_edge(ClaimEdge(
        source_id="contra",
        target_id="thesis",
        relation=RelationType.CONTRADICTS,
        weight=0.8,
    ))
    
    propagate_belief(graph)
    
    thesis_node = graph.get_node("thesis")
    assert thesis_node is not None
    assert thesis_node.propagated_belief is not None
    
    # Support: 1.0 * 0.6 * 1 = +0.6
    # Contra:  2.0 * 0.8 * -1 = -1.6
    # Net: 0.6 - 1.6 = -1.0
    assert thesis_node.propagated_belief.log_odds < 0.0
    
    # The stronger contradiction wins
    assert thesis_node.propagated_belief.probability < 0.5


def test_chain_propagation():
    """
    Test that belief propagates through a chain of claims.
    
    Setup:
    - A --SUPPORTS--> B --SUPPORTS--> C
    - A has strong belief, B and C start neutral
    
    Expected: C should receive indirect support from A through B.
    """
    graph = ClaimContextGraph()
    
    claim_a = ClaimNode(
        claim_id="a",
        text="Primary evidence",
        role=ClaimRole.SUPPORT,
        local_belief=BeliefState(log_odds=2.0, confidence=0.9),
    )
    
    claim_b = ClaimNode(
        claim_id="b",
        text="Intermediate claim",
        role=ClaimRole.SUPPORT,
        local_belief=BeliefState(log_odds=0.0, confidence=0.5),
    )
    
    claim_c = ClaimNode(
        claim_id="c",
        text="Final thesis",
        role=ClaimRole.THESIS,
        local_belief=BeliefState(log_odds=0.0, confidence=0.5),
    )
    
    graph.add_node(claim_a)
    graph.add_node(claim_b)
    graph.add_node(claim_c)
    
    graph.add_edge(ClaimEdge(
        source_id="a",
        target_id="b",
        relation=RelationType.SUPPORTS,
        weight=0.8,
    ))
    
    graph.add_edge(ClaimEdge(
        source_id="b",
        target_id="c",
        relation=RelationType.SUPPORTS,
        weight=0.7,
    ))
    
    propagate_belief(graph)
    
    # B should have received A's influence
    node_b = graph.get_node("b")
    assert node_b is not None
    assert node_b.propagated_belief is not None
    # B: 0.0 + (2.0 * 0.8) = 1.6
    assert node_b.propagated_belief.log_odds > 1.0
    
    # C should have received B's (now positive) influence
    node_c = graph.get_node("c")
    assert node_c is not None
    assert node_c.propagated_belief is not None
    # C: 0.0 + (1.6 * 0.7) = 1.12
    assert node_c.propagated_belief.log_odds > 0.5


def test_neutral_source_no_influence():
    """
    Test that a neutral source (log_odds=0) exerts no influence.
    
    Setup:
    - A: Neutral (log_odds=0)
    - B: Neutral
    - A --SUPPORTS--> B
    
    Expected: B remains neutral.
    """
    graph = ClaimContextGraph()
    
    neutral_a = ClaimNode(
        claim_id="a",
        text="Unverified claim",
        role=ClaimRole.BACKGROUND,
        local_belief=BeliefState(log_odds=0.0, confidence=0.3),
    )
    
    neutral_b = ClaimNode(
        claim_id="b",
        text="Thesis",
        role=ClaimRole.THESIS,
        local_belief=BeliefState(log_odds=0.0, confidence=0.5),
    )
    
    graph.add_node(neutral_a)
    graph.add_node(neutral_b)
    
    graph.add_edge(ClaimEdge(
        source_id="a",
        target_id="b",
        relation=RelationType.SUPPORTS,
        weight=1.0,
    ))
    
    propagate_belief(graph)
    
    node_b = graph.get_node("b")
    assert node_b is not None
    assert node_b.propagated_belief is not None
    
    # 0.0 * 1.0 * 1 = 0.0 influence
    # B stays at 0.0
    assert abs(node_b.propagated_belief.log_odds) < 0.01
