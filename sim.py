import os
import sys
import asyncio
import traceback

# Add to path
sys.path.insert(0, os.path.abspath('.'))

try:
    from spectrue_core.pipeline.factory import PipelineFactory
    from spectrue_core.runtime_config import SpectrueConfig
    from spectrue_core.pipeline.progress import ProgressEstimator

    class DummyAgent:
        pass

    class DummySearch:
        pass

    factory = PipelineFactory(search_mgr=DummySearch(), agent=DummyAgent())
    config = SpectrueConfig()

    async def main():
        for mode in ["general", "deep", "deep_v2"]:
            print(f"--- Simulating mode: {mode} ---")
            pipeline = factory.build(mode, config=config)
            
            async def dummy_callback(_event):
                pass
                
            estimator = ProgressEstimator(dummy_callback)
            # Use set_planned_nodes instead of on_dag_planned (outdated name probably)
            estimator.set_planned_nodes(pipeline.nodes)
            
            for node in pipeline.nodes:
                step = node.step
                name = getattr(step, "name", step.__class__.__name__)
                # Simulate engine.py emitting node_started
                await estimator.on_step_start(name)

    if __name__ == "__main__":
        asyncio.run(main())

except Exception:
    traceback.print_exc()
