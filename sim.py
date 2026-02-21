import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath('..'))

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
            
            async def dummy_callback(event):
                pass
                
            estimator = ProgressEstimator(dummy_callback)
            estimator.on_dag_planned(pipeline.nodes)
            
            for node in pipeline.nodes:
                step = node.step
                name = getattr(step, "name", step.__class__.__name__)
                # Simulate engine.py emitting node_started
                await estimator.on_step_start(name)

    asyncio.run(main())

except Exception as e:
    import traceback
    traceback.print_exc()
