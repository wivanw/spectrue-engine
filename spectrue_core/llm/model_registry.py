from enum import Enum

class ModelID(str, Enum):
    """Canonical model identifiers for internal engine routing."""
    
    # Cheap / Fast tier
    NANO = "gpt-5-nano"

    # Mid tier (balance of reasoning and cost)
    MID = "deepseek-chat"

    # High / Pro tier (complex reasoning, high reliability)
    PRO = "gpt-5.2"

# Backward compatibility aliases (to avoid breaking existing imports immediately)
MODEL_NANO = ModelID.NANO
MODEL_MID = ModelID.MID
MODEL_PRO = ModelID.PRO
