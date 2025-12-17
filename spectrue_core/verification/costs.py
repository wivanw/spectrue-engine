"""
Cost constants for verification operations.
"""

# Model costs (credits per claim analysis)
MODEL_COSTS = {
    "gpt-5-nano": 5,
    "gpt-5-mini": 20,
    "gpt-5.2": 100,
}

# Search costs (credits per search operation)
SEARCH_COSTS = {
    "basic": 80,
    "advanced": 160,
}
