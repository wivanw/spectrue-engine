import os
import sys
import importlib
import inspect
from pathlib import Path

# Adjust path to root before importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spectrue_core.pipeline.core import Step

steps_dir = Path(__file__).parent.parent / 'spectrue_core/pipeline/steps'
for p in steps_dir.rglob('*.py'):
    if p.name == '__init__.py':
        continue
    
    # Calculate relative module name
    rel_path = p.relative_to(steps_dir.parent.parent)
    module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')

    try:
        mod = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Step) and obj is not Step:
                if not getattr(obj, "name", None) or getattr(obj, "weight", None) is None:
                    print(f"Missing attributes in {name}: name={getattr(obj, 'name', None)}, weight={getattr(obj, 'weight', None)}")
    except Exception:
        pass
