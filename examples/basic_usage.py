"""
Basic Usage Example

This example demonstrates the simplest way to use Spectrue Engine
to verify a text claim.
"""

import asyncio
from spectrue_core.engine import SpectrueEngine
from spectrue_core.config import SpectrueConfig


async def main():
    # Configure the engine
    config = SpectrueConfig(
        openai_api_key="sk-your-key-here",
        tavily_api_key="tvly-your-key-here",
    )
    
    # Initialize engine
    engine = SpectrueEngine(config)
    
    # Analyze a claim
    text = """
    Breaking News: Scientists at NASA have announced the discovery 
    of a new moon orbiting Earth, which they have named "Petite".
    """
    
    print("Analyzing claim...")
    result = await engine.analyze_text(
        text=text,
        lang="en",
        mode="deep"
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)
    
    print("\n📊 Scores:")
    print(f"  Verified:       {result.get('verified_score', 0):.2f}")
    print(f"  Danger:         {result.get('danger_score', 0):.2f}")
    print(f"  Style:          {result.get('style_score', 0):.2f}")
    print(f"  Explainability: {result.get('explainability_score', 0):.2f}")
    
    print(f"\n📝 Verdict: {result.get('verdict', 'N/A')}")
    
    print("\n📖 Rationale:")
    print(f"  {result.get('rationale', 'No rationale provided')}")
    
    print(f"\n🔗 Sources ({len(result.get('sources', []))} found):")
    for source in result.get('sources', [])[:3]:
        print(f"  - {source.get('title', 'Untitled')}")
        print(f"    {source.get('url', '')}")


if __name__ == "__main__":
    asyncio.run(main())
