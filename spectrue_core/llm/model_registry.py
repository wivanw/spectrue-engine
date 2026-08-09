from enum import Enum

class ModelID(str, Enum):
    """Canonical model identifiers for internal engine routing.

    Kept current as of 2026-08. Previous generation (retired / deprecated):
      - gpt-5-nano      -> gpt-5.6-luna   (deprecated, shutdown 2026-12-11)
      - gpt-5.2         -> gpt-5.6-sol    (chat-latest snapshot shutdown 2026-08-10)
      - deepseek-chat / deepseek-reasoner  (aliases RETIRED 2026-07-24)

    Note on MID: DeepSeek's own successor for the `deepseek-chat` alias is
    deepseek-v4-flash, but that is a smaller model (284B total / 13B active vs
    the V3-class ~671B / 37B) and measurably weaker on reasoning-intensive and
    factual-recall subsets. MID carries claim extraction, stance/clustering and
    mid-tier judging here, so we map it to deepseek-v4-pro (1.6T / 49B) to keep
    capability parity. It is still ~11x cheaper on input and ~34x on output
    than PRO, so the tier keeps its cost-saving purpose.
    """

    # Cheap / Fast tier
    NANO = "gpt-5.6-luna"

    # Mid tier (balance of reasoning and cost)
    MID = "deepseek-v4-pro"

    # High / Pro tier (complex reasoning, high reliability)
    PRO = "gpt-5.6-sol"

# Backward compatibility aliases (to avoid breaking existing imports immediately)
MODEL_NANO = ModelID.NANO
MODEL_MID = ModelID.MID
MODEL_PRO = ModelID.PRO
