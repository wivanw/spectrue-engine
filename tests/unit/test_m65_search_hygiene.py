import pytest
from unittest.mock import AsyncMock
from spectrue_core.verification.search_mgr import SearchManager
from spectrue_core.config import SpectrueConfig

@pytest.fixture
def mock_config():
    # Pass api key to constructor
    cfg = SpectrueConfig(tavily_api_key="test-key")
    return cfg

@pytest.fixture
def search_manager(mock_config):
    return SearchManager(mock_config)

@pytest.mark.asyncio
async def test_filter_search_results(search_manager):
    # Setup test data
    results = [
        {"title": "Good News", "link": "https://bbc.com/news/1", "relevance_score": 0.9, "snippet": "ok"},
        {"title": "Low Score", "link": "https://example.com/bad", "relevance_score": 0.10, "snippet": "low"},
        {"title": "Reddit", "link": "https://reddit.com/r/news", "relevance_score": 0.8, "snippet": "junk"},
        {"title": "Bad Ext", "link": "https://example.com/file.txt", "relevance_score": 0.8, "snippet": "txt"},
    ]
    
    # Test "news" intent (Should filter low score and extensions, but allow reddit if score is high)
    # Rationale: We rely on Tavily topic="news" to not return reddit. If it returns it AND score is high, we take it.
    filtered_news = search_manager._filter_search_results(results, intent="news")
    
    # Validation
    urls_news = [r["link"] for r in filtered_news]
    assert "https://bbc.com/news/1" in urls_news
    assert "https://example.com/bad" not in urls_news  # Score < 0.15
    assert "https://example.com/file.txt" not in urls_news # Extension
    # Reddit remains if score is high (0.8), because we removed the hardcoded blocklist
    assert "https://reddit.com/r/news" in urls_news 
    assert len(filtered_news) == 2

@pytest.mark.asyncio
async def test_search_unified_flow(search_manager):
    # Mock web_tool.search
    mock_results = [
        {"title": "T1", "link": "https://bbc.com", "relevance_score": 0.9, "snippet": "t1"},
        {"title": "T2", "link": "https://reddit.com", "relevance_score": 0.9, "snippet": "t2"}, # Junk for news
        {"title": "T3", "link": "https://low.com", "relevance_score": 0.1, "snippet": "low"}, # Low score
    ]
    search_manager.web_tool.search = AsyncMock(return_value=("fake_context", mock_results))
    
    # Execute
    ctx, res = await search_manager.search_unified("query", intent="news")
    
    # Verify filter applied (Low score removed, Reddit retained)
    assert len(res) == 2
    links = [r["link"] for r in res]
    assert "https://bbc.com" in links
    assert "https://reddit.com" in links
    
    # Verify context reconstruction
    assert "T1" in ctx
    assert "T2" in ctx
    assert "low" not in ctx
