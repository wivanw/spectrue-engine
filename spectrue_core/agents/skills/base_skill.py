from spectrue_core.agents.llm_client import LLMClient
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import EngineRuntimeConfig
import logging

logger = logging.getLogger(__name__)

class BaseSkill:
    def __init__(self, config: SpectrueConfig, llm_client: LLMClient):
        self.config = config
        self.runtime = (config.runtime if config else None) or EngineRuntimeConfig.load_from_env()
        self.llm_client = llm_client
