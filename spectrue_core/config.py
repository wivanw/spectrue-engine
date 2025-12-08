# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

from typing import Optional
from pydantic import BaseModel, Field

class SpectrueConfig(BaseModel):
    """
    Configuration for the Spectrue Core Engine.
    Decouples the engine from environment variables.
    """
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(None, description="OpenAI API Key for LLM operations")
    openai_model: str = Field("gpt-4o", description="Model to use for main analysis")
    
    # Search Configuration
    tavily_api_key: Optional[str] = Field(None, description="Tavily API Key for web search")
    google_search_api_key: Optional[str] = Field(None, description="Google Custom Search API Key (optional)")
    google_search_cse_id: Optional[str] = Field(None, description="Google Custom Search Engine ID (optional)")
    
    # RAG Configuration
    gcs_bucket_name: Optional[str] = Field(None, description="Google Cloud Storage bucket for RAG indexes")
    project_root: Optional[str] = Field(None, description="Root path for local RAG data storage")
    
    # Verification Settings
    min_confidence_threshold: float = Field(0.7, description="Minimum confidence to verify a claim")
    max_search_depth: int = Field(3, description="Maximum recursion depth for deep verification")

    # Local LLM (Optional)
    local_llm_url: Optional[str] = Field(None, description="URL for local LLM (e.g., Llama 3 via llama-cpp-python)")
    
    class Config:
        env_prefix = "SPECTRUE_"
