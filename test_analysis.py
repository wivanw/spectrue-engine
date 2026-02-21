import asyncio
import os
import sys
import logging

sys.path.insert(0, os.path.abspath('..'))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("spectrue_core.pipeline.progress")
logger.setLevel(logging.WARNING)

from spectrue_core.runtime_config import SpectrueConfig
from spectrue_core.engine import SpectrueEngine
from spectrue_core.pipeline.progress import ProgressEvent

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
        await engine.verify(
            text=test_text,
            lang="en",
            mode="deep_v2",
            progress_callback=progress_callback
        )
    except Exception as e:
        print(f"Failed with {e}")

asyncio.run(main())
