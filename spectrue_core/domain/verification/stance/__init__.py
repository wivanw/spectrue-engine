"""Stance domain package."""

from .model import StanceFeatures, StancePosterior
from spectrue_core.verification.scoring.stance_posterior import (
    compute_stance_posterior,
    source_prior_from_tier,
)

__all__ = [
    "StanceFeatures",
    "StancePosterior",
    "compute_stance_posterior",
    "source_prior_from_tier",
]
