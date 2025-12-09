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
    
    # Verification Settings
    min_confidence_threshold: float = Field(0.7, description="Minimum confidence to verify a claim")
    max_search_depth: int = Field(3, description="Maximum recursion depth for deep verification")

    # Local LLM (Optional)
    local_llm_url: Optional[str] = Field(None, description="URL for local LLM (e.g., Llama 3 via llama-cpp-python)")
    
    class Config:
        env_prefix = "SPECTRUE_"
