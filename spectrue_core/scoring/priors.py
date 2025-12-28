"""Legacy scoring priors (deprecated).

Spectrue policy: Source Tier must NOT bias factual veracity (G).
Tier affects RGBA Alpha (Explainability) only.

This module previously provided a Tier-based prior in log-odds space.
It is kept temporarily for backward compatibility with older experiments,
but it is intentionally NOT re-exported from :mod:`spectrue_core.scoring`.
"""

from __future__ import annotations

from typing import Dict

import warnings

# Tier Baselines from Research
# Log-odds interpretation: 0.0 = 50% probability
TIER_BASE_LOG_ODDS: Dict[int, float] = {
    1: 1.0,   # Scientific/Official (~73%)
    2: 0.5,   # High Quality News (~62%)
    3: 0.0,   # General (50%)
    4: -1.0,  # Low Quality/User (~27%)
}

# Alpha coefficient for Brand Trust (0-100)
# Trust of 100 adds 1.0 log-odds
# Trust of 0 subtracts 1.0 log-odds
TRUST_ALPHA = 0.02

def calculate_prior(tier: int, brand_trust: float) -> float:
    """
    DEPRECATED.

    Calculates a prior log-odds based on Source Tier and Brand Trust.
    Formula: Base(Tier) + Alpha * (Trust - 50)
    
    Args:
        tier: Source tier (1-4).
        brand_trust: Trust score (0-100).
        
    Returns:
        Initial belief in log-odds space.
    """
    warnings.warn(
        "calculate_prior() is deprecated: Tier must not bias veracity (G). "
        "Use tier priors only for explainability (A).",
        DeprecationWarning,
        stacklevel=2,
    )
    # Default to Tier 3 (Neutral) if tier is invalid/unknown
    base = TIER_BASE_LOG_ODDS.get(tier, 0.0) 
    
    trust_modifier = TRUST_ALPHA * (brand_trust - 50.0)
    
    return base + trust_modifier
