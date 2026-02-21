import os
import sys
import traceback

sys.path.insert(0, os.path.abspath('..'))
try:
    from spectrue_core.pipeline.factory import PipelineFactory
    from spectrue_core.runtime_config import SpectrueConfig

    class DummyAgent:
        pass

    class DummySearch:
        pass

    factory = PipelineFactory(search_mgr=DummySearch(), agent=DummyAgent())
    config = SpectrueConfig()

    modes = ["general", "deep", "deep_v2"]
    for mode in modes:
        try:
            pipeline = factory.build(mode, config=config)
            for node in pipeline.nodes:
                step = node.step
                name = getattr(step, "name", getattr(step.__class__, "name", None))
                weight = getattr(step, "weight", getattr(step.__class__, "weight", None))
                if name is None or weight is None:
                    print(f"Mode {mode}: Step {step.__class__.__name__} missing - name={name}, weight={weight}")
                    
        except Exception as e:
            print(f"Mode {mode} failed: {e}")
            traceback.print_exc()
except Exception as e:
    print(f"Import failed: {e}")
    traceback.print_exc()

