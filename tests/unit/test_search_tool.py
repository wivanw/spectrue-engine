
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from spectrue_core.tools.search_tool import WebSearchTool
from spectrue_core.tools.google_cse_search import GoogleCSESearchTool

@pytest.mark.unit
class TestWebSearchTool:
    
    @pytest.fixture
    def tool(self, mock_config):
        with patch("spectrue_core.tools.tavily_client.httpx.AsyncClient") as mock_client_cls, \
             patch("spectrue_core.tools.cache_utils.diskcache.Cache") as mock_cache_cls:
            
            # Setup HTTP Client Mock
            mock_http_client = AsyncMock()
            mock_client_cls.return_value = mock_http_client
            
            # Setup Cache Mock (Always Miss)
            mock_cache = MagicMock()
            mock_cache.__contains__.return_value = False
            mock_cache.get.return_value = None
            mock_cache_cls.return_value = mock_cache

            tool = WebSearchTool(config=mock_config)
            tool.client = mock_http_client
            tool.cache = mock_cache
            tool.page_cache = mock_cache
            
            yield tool
            
    @pytest.mark.asyncio
    async def test_search_basic(self, tool):
        # Prepare the Response object with 3 unique domains to avoid M61 diversification pass
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/test",
                    "content": "This is a test content snippet.",
                    "score": 0.9
                },
                {
                    "title": "Test Result 2",
                    "url": "https://another.org/page",
                    "content": "Another test content snippet.",
                    "score": 0.85
                },
                {
                    "title": "Test Result 3",
                    "url": "https://third-domain.net/article",
                    "content": "Third test content snippet.",
                    "score": 0.8
                }
            ]
        }
        # Configure client.post to return this response
        tool.client.post.return_value = mock_response

        context, sources = await tool.search("test query")
        
        assert len(sources) == 3
        assert sources[0]["title"] == "Test Result 1"
        assert sources[0]["relevance_score"] > 0.0
        
        # Updated assertions to match actual transformation in search_tool.py
        assert "link" in sources[0] 
        assert "snippet" in sources[0]
        assert sources[0]["origin"] == "WEB"

    @pytest.mark.asyncio
    async def test_extract_text(self, tool):
        # Prepare response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "url": "https://example.com/article",
                    "raw_content": "Simple text content " * 10,
                    "content": "Simple text content " * 10
                }
            ]
        }
        tool.client.post.return_value = mock_response
        
        text = await tool._fetch_extract_text("https://example.com/article")

        # Verify call was made
        tool.client.post.assert_called()
        
        assert text is not None
        assert "Simple text content" in text

    def test_clean_results_deduplication(self, tool):
        raw_results = [
            {"title": "A", "url": "http://a.com", "content": "foo"},
            {"title": "a", "url": "http://a.com", "content": "bar"}, 
            {"title": "B", "url": "http://b.com", "content": "baz"},
        ]
        cleaned = tool._clean_results(raw_results)
        assert len(cleaned) == 2
        titles = {r["title"] for r in cleaned}
        assert "A" in titles
        assert "B" in titles

    def test_rank_and_filter_logic(self, tool):
        results = [
            {"title": "Exact Match", "url": "http://1.com", "content": "The quick brown fox"},
            {"title": "Unrelated", "url": "http://2.com", "content": "XXX YYY ZZZ"},
        ]
        ranked = tool._rank_and_filter("quick brown fox", results)
        assert len(ranked) >= 1
        assert ranked[0]["title"] == "Exact Match"

    def test_tavily_score_blending(self, tool):
        long_content = "Some Content " * 10 
        res = tool._relevance_score(
            query="irrelevant query terms",
            title="Some Title",
            content=long_content,
            url="http://x.com",
            tavily_score=0.95
        )
        assert 0.6 < res < 0.8

        res2 = tool._relevance_score(
            query="test",
            title="test",
            content="test",
            url="http://x.com",
            tavily_score=0.5
        )
        assert res2 > 0.9

    # =============================================================================
    # M61: New tests for Search Quality v2
    # =============================================================================

    def test_archive_domain_filtered(self, tool):
        """T5: Archive domains like un.org PDFs should be filtered unless they have news keywords."""
        results = [
            {"title": "UN Document", "url": "https://digitallibrary.un.org/record/12345.pdf", "content": "Legal document about treaties"},
            {"title": "Reuters News", "url": "https://reuters.com/news/article", "content": "Breaking news about treaties"},
        ]
        ranked = tool._rank_and_filter("treaties international law", results)
        
        # UN digital library PDF should be filtered (no news keywords)
        urls = [r["url"] for r in ranked]
        assert "https://digitallibrary.un.org/record/12345.pdf" not in urls
        assert "https://reuters.com/news/article" in urls

    def test_archive_domain_kept_with_news_keyword(self, tool):
        """T5: Archive domains should be kept if they contain news keywords like '2025'."""
        results = [
            {"title": "UN Report 2025", "url": "https://docs.un.org/record/2025/report.pdf", "content": "Breaking 2025 report about sanctions"},
        ]
        ranked = tool._rank_and_filter("UN sanctions 2025", results)
        
        # Should be kept because content has "2025" and "breaking"
        assert len(ranked) >= 1

    def test_archive_url_pattern_filtered(self, tool):
        """T5: URLs with archive patterns like /publications/ should be filtered."""
        results = [
            {"title": "Old Publication", "url": "https://example.org/publications/old-report", "content": "Historical data"},
            {"title": "News Article", "url": "https://example.org/news/current", "content": "Historical data current events"},
        ]
        ranked = tool._rank_and_filter("historical data", results)
        
        # /publications/ URL should be filtered
        urls = [r["url"] for r in ranked]
        assert "https://example.org/publications/old-report" not in urls

    def test_date_freshness_boost_current_year(self, tool):
        """T5: URLs with current year should get +0.08 boost."""
        import datetime
        current_year = datetime.datetime.now().year
        
        # URL with current year
        score_fresh = tool._relevance_score(
            query="test news",
            title="Test News",
            content="Test news content here",
            url=f"https://news.com/{current_year}/12/article",
            tavily_score=None
        )
        
        # URL without year
        score_no_year = tool._relevance_score(
            query="test news",
            title="Test News",
            content="Test news content here",
            url="https://news.com/article",
            tavily_score=None
        )
        
        # Fresh URL should score higher
        assert score_fresh > score_no_year

    def test_date_freshness_penalty_old_year(self, tool):
        """T5: URLs with old years (>2 years ago) should get -0.12 penalty."""
        import datetime
        old_year = datetime.datetime.now().year - 3
        
        # URL with old year
        score_old = tool._relevance_score(
            query="test news",
            title="Test News",
            content="Test news content here",
            url=f"https://news.com/{old_year}/12/article",
            tavily_score=None
        )
        
        # URL without year
        score_no_year = tool._relevance_score(
            query="test news",
            title="Test News",
            content="Test news content here",
            url="https://news.com/article",
            tavily_score=None
        )
        
        # Old URL should score lower
        assert score_old < score_no_year

@pytest.mark.unit
class TestGoogleCSESearchTool:

    @pytest.fixture
    def tool(self, mock_config):
        with patch("spectrue_core.tools.google_cse_search.httpx.AsyncClient") as mock_client_cls, \
             patch("spectrue_core.tools.google_cse_search.diskcache.Cache") as mock_cache_cls:
            
            mock_http_client = AsyncMock()
            mock_client_cls.return_value = mock_http_client
            
            mock_cache = MagicMock()
            mock_cache.__contains__.return_value = False
            mock_cache_cls.return_value = mock_cache
            
            tool = GoogleCSESearchTool(config=mock_config)
            tool.client = mock_http_client
            tool.cache = mock_cache
            yield tool

    @pytest.mark.asyncio
    async def test_search_cse(self, tool):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "Google Result",
                    "link": "https://google.com/res",
                    "snippet": "Snippet from Google"
                }
            ]
        }
        tool.client.get.return_value = mock_response

        context, sources = await tool.search("query")
        
        assert "Snippet from Google" in context
        assert len(sources) == 1
        assert sources[0]["provider"] == "google_cse"

    @pytest.mark.asyncio
    async def test_disabled_tool(self, tool):
        tool.api_key = None
        context, sources = await tool.search("query")
        assert context == ""
        assert sources == []
