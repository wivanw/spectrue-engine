import pytest


from spectrue_core.pipeline.progress import ProgressEstimator, ProgressEvent
from spectrue_core.pipeline.dag import StepNode

class MockStep:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

@pytest.mark.asyncio
async def test_progress_estimator_monotonicity():
    events = []
    async def callback(event: ProgressEvent):
        events.append(event)
    
    estimator = ProgressEstimator(callback)
    
    # 1. Define plan
    steps = [
        MockStep("metering", 5.0),
        MockStep("search", 20.0),
        MockStep("judge", 50.0),
        MockStep("finish", 5.0),
    ]
    nodes = [StepNode(step=s) for s in steps]
    estimator.set_planned_nodes(nodes)
    
    last_percent = 0
    
    for node in nodes:
        name = node.name
        # Start
        await estimator.on_step_start(name)
        assert len(events) > 0
        current_event = events[-1]
        
        # Verify status key mapping is automatically generated
        assert current_event.status_key == f"loader.{name}"
        
        # Verify percent monotonicity
        assert current_event.percent >= last_percent
        
        # End
        await estimator.on_step_end(name)
        
        # Check internal state matches completed weight
        index = nodes.index(node)
        expected_weight = sum(n.step.weight for n in nodes[:index+1])
        assert estimator.completed_weight == expected_weight
        
        last_percent = events[-1].percent


@pytest.mark.asyncio
async def test_progress_estimator_unknown_steps():
    events = []
    async def callback(event: ProgressEvent):
        events.append(event)
    
    estimator = ProgressEstimator(callback)
    
    # Empty plan
    estimator.set_planned_nodes([])
    
    await estimator.on_step_start("unknown_step")
    # Even for unknown steps, it generates "loader.unknown_step"
    assert events[-1].status_key == "loader.unknown_step"
    
    await estimator.on_step_end("unknown_step")
    # Step object missing, weight fallback to 0.0 in on_step_end
    assert estimator.completed_weight == 0.0
