import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from spectrue_core.pipeline.core import Step

import importlib
import inspect
from pathlib import Path

steps_dir = Path('../spectrue_core/pipeline/steps')
for p in steps_dir.rglob('*.py'):
    if p.name == '__init__.py':
        continue
    module_name = 'spectrue_core.pipeline.steps.' + p.stem
    try:
        mod = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Step) and obj is not Step:
                has_name = hasattr(obj, 'name') or 'name' in obj.__annotations__ or 'name' in obj.__dict__
                has_weight = hasattr(obj, 'weight') or 'weight' in obj.__annotations__ or 'weight' in obj.__dict__
                if not getattr(obj, "name", None) or getattr(obj, "weight", None) is None:
                    print(f"Missing attributes in {name}: name={getattr(obj, 'name', None)}, weight={getattr(obj, 'weight', None)}")
    except Exception as e:
        pass
