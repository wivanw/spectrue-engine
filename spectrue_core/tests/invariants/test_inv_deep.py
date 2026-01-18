import pytest
from spectrue_core.graph.propagation import propagate_belief
from spectrue_core.graph.context import ClaimContextGraph
from spectrue_core.schema.scoring import ClaimNode, ClaimEdge, BeliefState, ClaimRole, RelationType

def test_inv_032_graph_propagation_bounds():
    """
    INV-032: Graph Propagation Bounds.
    Normative: Propagation MUST NEVER flip a claim (increase confidence/verification status)
    in the absence of direct evidence.
    
    This test verifies that while log_odds (belief) propagates from strong neighbors,
    the 'confidence' metric (proxy for direct evidence weight) remains unchanged (low).
    """
    graph = ClaimContextGraph()
    
    # 1. Parent Node (Strongly verified)
    parent_id = "parent"
    parent_node = ClaimNode(
        claim_id=parent_id,
        text="Parent verified claim",
        role=ClaimRole.THESIS,
        local_belief=BeliefState(log_odds=5.0, confidence=0.9)  # High confidence
    )
    graph.add_node(parent_node)
    
    # 2. Child Node (No Evidence / Neutral)
    child_id = "child"
    child_node = ClaimNode(
        claim_id=child_id,
        text="Child unverified claim",
        role=ClaimRole.SUPPORT,
        local_belief=BeliefState(log_odds=0.0, confidence=0.0)  # Zero confidence/evidence
    )
    graph.add_node(child_node)
    
    # 3. Edge (Parent SUPPORTS Child)
    edge = ClaimEdge(
        source_id=parent_id,
        target_id=child_id,
        relation=RelationType.SUPPORTS,
        weight=1.0  # Strong semantic link
    )
    graph.add_edge(edge)
    
    # Run propagation
    propagate_belief(graph)
    
    # Check results on Child
    updated_child = graph.get_node(child_id)
    assert updated_child.propagated_belief is not None
    
    # A. Belief should increase (log_odds > 0) due to parent support
    # (5.0 * 1.0 = 5.0 added to 0.0)
    assert updated_child.propagated_belief.log_odds > 0.0
    assert updated_child.propagated_belief.log_odds == 5.0
    
    # B. Confidence MUST NOT increase (Invariant Violation check)
    # It must remain at the local level (0.0)
    assert updated_child.propagated_belief.confidence == 0.0
    
    # This ensures that "Verified" status (typically requiring high G + high Confidence/A)
    # cannot be synthesized purely from neighbors.

@pytest.mark.asyncio
async def test_inv_030_deep_structure_compliance():
    """
    INV-030: Structural Consistency.
    Verify that Deep Mode output adheres to ClaimResult contract.
    """
    # This involves verifying AssembleDeepResultStep output structure
    from spectrue_core.pipeline.steps.deep_claim import AssembleDeepResultStep, DeepClaimContext
    from spectrue_core.pipeline.core import PipelineContext
    from spectrue_core.pipeline.mode import DEEP_MODE
    from spectrue_core.schema.claim_frame import ClaimFrame, JudgeOutput, RGBAScore, ContextExcerpt, ContextMeta

    frame = ClaimFrame(
        claim_id="c1",
        claim_text="Structure Test",
        claim_language="en",
        context_excerpt=ContextExcerpt(text="ctx"),
        context_meta=ContextMeta(document_id="d1"),
        evidence_items=()
    )
    
    judge_output = JudgeOutput(
        claim_id="c1",
        rgba=RGBAScore(r=0.1, g=0.1, b=0.1, a=0.1),
        confidence=0.8,
        verdict="Supported",
        explanation="Exp",
        sources_used=("http://a.com",)
    )
    
    deep_ctx = DeepClaimContext(
        claim_frames=[frame],
        judge_outputs={"c1": judge_output}
    )
    
    ctx = PipelineContext(mode=DEEP_MODE, extras={"deep_claim_ctx": deep_ctx})
    step = AssembleDeepResultStep()
    result_ctx = await step.run(ctx)
    deep_res = result_ctx.get_extra("deep_analysis_result")
    
    # Verify Schema
    assert deep_res is not None
    assert "claim_results" in deep_res
    res = deep_res["claim_results"][0]
    
    # Required keys per contract
    required_keys = ["claim_id", "status", "rgba", "verdict_score", "explanation", "sources_used"]
    for k in required_keys:
        assert k in res, f"Missing key {k}"

