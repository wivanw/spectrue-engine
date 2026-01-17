"""
Registry of supported LLM models for routing and billing.

These constants serve as canonical IDs for models used throughout the engine,
allowing central management of backend model identifiers mapping to pricing.
"""

# Cheap / Fast tier
MODEL_NANO = "gpt-5-nano"

# Mid tier (balance of reasoning and cost)
MODEL_MID = "deepseek-chat"

# High / Pro tier (complex reasoning, high reliability)
MODEL_PRO = "gpt-5.2"
