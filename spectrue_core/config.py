# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
from typing import Optional

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from spectrue_core.runtime_config import EngineRuntimeConfig

class SpectrueConfig(BaseModel):
    """
    Configuration for the Spectrue Core Engine.
    Decouples the engine from environment variables.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, env_prefix="SPECTRUE_")

    # LLM Configuration
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"), description="OpenAI API Key for LLM operations")

    # Search Configuration
    # Search Configuration
    tavily_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"), description="Tavily API Key for web search")
    google_search_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_SEARCH_API_KEY"), description="Google Custom Search API Key (optional)")
    google_search_cse_id: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_SEARCH_CSE_ID"), description="Google Custom Search Engine ID (optional)")
    google_fact_check_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_FACT_CHECK_KEY"), description="Google Fact Check Tools API Key (optional)")

    # Verification Settings
    min_confidence_threshold: float = Field(0.7, description="Minimum confidence to verify a claim")
    max_search_depth: int = Field(3, description="Maximum recursion depth for deep verification")

    # Local LLM (Optional)
    local_llm_url: Optional[str] = Field(None, description="URL for local LLM (e.g., Llama 3 via llama-cpp-python)")

    # Runtime configuration (loaded from env once; excludes secrets)
    runtime: EngineRuntimeConfig = Field(default_factory=EngineRuntimeConfig.load_from_env)
