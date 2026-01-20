import pytest


from spectrue_core.pipeline.progress import ProgressEstimator, ProgressEvent, STATUS_KEYS, STEP_WEIGHTS

@pytest.mark.asyncio
async def test_progress_estimator_monotonicity():
    events = []
    async def callback(event: ProgressEvent):
        events.append(event)
    
    estimator = ProgressEstimator(callback)
    
    # Simulate a sequence of steps
    steps = [
        "metering_setup",
        "extract_claims",
        "build_queries",
        "search_execution",
        "process_search_results",
        "clustering",
        "select_evidence",
        "summarize_evidence",
        "judge_claims",
        "assemble_result"
    ]
    
    last_percent = 0
    
    for step in steps:
        # Start
        await estimator.on_step_start(step)
        assert len(events) > 0
        current_event = events[-1]
        
        # Verify status key mapping
        expected_key = STATUS_KEYS.get(step, "status.processing")
        assert current_event.status_key == expected_key
        
        # Verify percent monotonicity (should not decrease)
        assert current_event.percent >= last_percent
        # Percent updates on step END logic, so on start it might be same as previous end
        
        # End
        await estimator.on_step_end(step)
        # Note: on_step_end updates internal state but does NOT emit event usually
        
        # Check internal state matches completed weight
        expected_weight = sum(STEP_WEIGHTS[s] for s in steps[:steps.index(step)+1])
        assert estimator.completed_weight == expected_weight
        
        # Verify percent calculation for next start
        # The next start call will use the new weight
        
        if len(events) > 1:
             last_percent = events[-1].percent


@pytest.mark.asyncio
async def test_progress_estimator_unknown_steps():
    events = []
    async def callback(event: ProgressEvent):
        events.append(event)
    
    estimator = ProgressEstimator(callback)
    
    await estimator.on_step_start("unknown_step")
    assert events[-1].status_key == "status.processing"
    
    await estimator.on_step_end("unknown_step")
    # Weight shouldn't change
    assert estimator.completed_weight == 0
