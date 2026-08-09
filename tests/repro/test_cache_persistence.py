
import asyncio
import os
import sys

# Add spectrue_core to path
sys.path.append(os.path.abspath(os.curdir))

from spectrue_core.tools.web_search_tool import WebSearchTool
from spectrue_core.config import SpectrueConfig

async def test_cache():
    print("--- Testing WebSearchTool Caching ---")
    config = SpectrueConfig()
    # Ensure tool can work without API key for cache testing
    tool = WebSearchTool(config)
    
    query = "test query " + str(os.urandom(4).hex())
    cache_key = tool._clean_url_key(f"http://example.com/{query}")
    
    print(f"Testing page cache with key: {cache_key}")
    tool._write_page_cache(cache_key, "Life is good")
    
    val = tool._try_page_cache(cache_key)
    print(f"Page Cache lookup: {val}")
    
    if val == "Life is good":
        print("✅ Page Cache works in memory/session")
    else:
        print("❌ Page Cache failed")

    # Check search cache
    search_key = "test_search_key"
    tool.cache.set(search_key, ("context", [{"url": "http://example.com"}]), expire=60)
    
    cached = tool.cache.get(search_key)
    print(f"Search Cache lookup: {cached}")
    
    if cached and cached[0] == "context":
        print("✅ Search Cache works in memory/session")
    else:
        print("❌ Search Cache failed")
        
    tool.close()
    
    # Re-open and check persistence
    tool2 = WebSearchTool(config)
    val2 = tool2._try_page_cache(cache_key)
    cached2 = tool2.cache.get(search_key)
    
    print(f"Persistence check - Page Cache: {val2}")
    print(f"Persistence check - Search Cache: {cached2}")
    
    if val2 == "Life is good" and cached2 and cached2[0] == "context":
        print("✅ Caching is PERSISTENT")
    else:
        print("❌ Caching is NOT PERSISTENT")
    
    tool2.close()

if __name__ == "__main__":
    asyncio.run(test_cache())
