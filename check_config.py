from spectrue_core.runtime_config import EngineRuntimeConfig
import os

try:
    print("Loading config...")
    cfg = EngineRuntimeConfig.load_from_env()
    print("Config loaded successfully!")
    print(f"trace_safe_payloads: {cfg.features.trace_safe_payloads}")
except Exception as e:
    print(f"FAILED to load config: {e}")
    exit(1)
