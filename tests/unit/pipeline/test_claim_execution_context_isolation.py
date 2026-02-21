import pytest
from spectrue_core.pipeline.claims.execution_context import ClaimExecutionContext
from spectrue_core.use_cases.verification.orchestration.execution_state import ClaimExecutionState


def test_claim_context_deep_copy_isolation():
    """Ensure mutable structured data is deep copied."""
    claim_dict = {"id": "c1", "text": "Claim text", "meta": {"key": "val"}}
    evidence = [{"url": "http://test.com", "score": 0.5}]
    
    state = ClaimExecutionState(claim_id="c1")
    state.phases_completed.add("extract")
    
    ctx1 = ClaimExecutionContext.create(
        claim=claim_dict,
        evidence_items=evidence,
        state=state
    )
    
    # Assert id differences to verify deep copy
    assert id(ctx1.claim) != id(claim_dict)
    assert id(ctx1.claim["meta"]) != id(claim_dict["meta"])
    
    assert id(ctx1.evidence_items) != id(evidence)
    assert id(ctx1.evidence_items[0]) != id(evidence[0])
    
    assert id(ctx1.state) != id(state)
    assert id(ctx1.state.phases_completed) != id(state.phases_completed)


def test_with_evidence_immutability():
    """Ensure with_evidence returns a new instance without mutating the old."""
    ctx1 = ClaimExecutionContext.create(claim={"id": "c1"})
    
    assert len(ctx1.evidence_items) == 0
    
    evidence = [{"url": "http://test.com"}]
    ctx2 = ctx1.with_evidence(evidence)
    
    assert ctx1 is not ctx2
    assert len(ctx1.evidence_items) == 0
    assert len(ctx2.evidence_items) == 1
    
    # Mutating original evidence should not affect ctx2
    evidence[0]["mutated"] = True
    assert "mutated" not in ctx2.evidence_items[0]


def test_with_state_update_immutability():
    """Ensure with_state_update returns a new instance."""
    state1 = ClaimExecutionState(claim_id="c1", phases_completed={"extract"})
    ctx1 = ClaimExecutionContext.create(claim={"id": "c1"}, state=state1)
    
    new_state = ClaimExecutionState(claim_id="c1", phases_completed={"extract", "judge"})
    ctx2 = ctx1.with_state_update(new_state)
    
    assert ctx1 is not ctx2
    assert "judge" not in ctx1.state.phases_completed
    assert "judge" in ctx2.state.phases_completed


def test_with_retrieval_plan_immutability():
    """Ensure with_retrieval_plan returns a new instance."""
    ctx1 = ClaimExecutionContext.create(claim={"id": "c1"})
    
    plan = {"query": "test query"}
    ctx2 = ctx1.with_retrieval_plan(plan)
    
    assert ctx1 is not ctx2
    assert ctx1.retrieval_plan == {}
    assert ctx2.retrieval_plan == plan
    
    plan["query"] = "mutated"
    assert ctx2.retrieval_plan["query"] == "test query"
