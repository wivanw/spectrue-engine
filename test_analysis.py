import asyncio
import os
import sys
import logging

# Move imports to top, but keep sys.path adjustment before them if needed. 
# Actually, E402 can be ignored if sys.path is manipulated, but here I can just fix it.
from spectrue_core.runtime_config import SpectrueConfig
from spectrue_core.engine import SpectrueEngine

# Removed unused ProgressEvent

sys.path.insert(0, os.path.abspath('.')) # Changed to . since it's likely running from root

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("spectrue_core.pipeline.progress")
logger.setLevel(logging.WARNING)

async def progress_callback(*args, **kwargs):
    pass

async def main():
    config = SpectrueConfig()
    config.openai_api_key = "dummy"
    config.tavily_api_key = "dummy"
    config.redis_url = None
    
    engine = SpectrueEngine(config=config)
    
    test_text = "The earth is flat and we never went to the moon."
    
    print("--- Running DEEP V2 ---")
    try:
        # verify() seems to be a method on engine, or is it analyze_text? 
        # Looking at engine.py, it's analyze_text.
        await engine.analyze_text(
            text=test_text,
            lang="en",
            analysis_mode="deep_v2",
            progress_callback=progress_callback
        )
    except Exception as e:
        print(f"Failed with {e}")

if __name__ == "__main__":
    asyncio.run(main())
