import pytest
from decimal import Decimal
from spectrue_core.verification.pipeline.pipeline_metering import create_metering_context, attach_cost_summary
from spectrue_core.billing.cost_event import CostEvent

@pytest.mark.asyncio
async def test_inv_070_spending_traceability():
    """
    INV-070: Traceability of Spend.
    Verify that every API response includes an audit trail of resources consumed,
    specifically token counts and model IDs.
    """
    metering = create_metering_context()
    
    # Simulate consumption
    event = CostEvent(
        stage="llm_generation",
        provider="openai",
        cost_usd=0.005,
        cost_credits=Decimal("5.0"),
        meta={
            "model": "gpt-4o",
            "tokens_in": 100,
            "tokens_out": 50
        }
    )
    metering.ledger.record_event(event)

    # Prepare result payload
    payload = {"status": "ok"}
    
    # Run attachment logic (typically done at end of pipeline)
    updated = attach_cost_summary(payload, metering=metering)
    
    # 1. Cost Summary Presence
    assert "cost_summary" in updated
    summary = updated["cost_summary"]
    
    # 2. Aggregates
    assert summary["total_credits"] == 5.0
    assert summary["total_usd"] == 0.005
    
    # 3. Granular Events (Traceability)
    assert len(summary["events"]) == 1
    ev_out = summary["events"][0]
    # Check meta fields where specific details reside
    assert ev_out["meta"]["model"] == "gpt-4o"
    assert ev_out["meta"]["tokens_in"] == 100
    assert ev_out["meta"]["tokens_out"] == 50
    
    # 4. Usage Ledger (Audit) presence
    # Check that audit.usage_ledger is populated as per attach_cost_summary
    assert "audit" in updated
    assert "usage_ledger" in updated["audit"]
    usage_ledger = updated["audit"]["usage_ledger"]
    assert "counts" in usage_ledger
    # Verify provider count updated (requires provider="openai" in at least one event)
    assert usage_ledger["counts"]["llm_calls_total"] == 1
