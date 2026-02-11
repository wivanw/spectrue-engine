"""Bayesian uncertainty and evidence features."""
import math
from typing import Any

# =============================================================================
# BAYESIAN UNCERTAINTY (BETA-BERNOULLI MODEL)
# =============================================================================
#
# We estimate: θ = "probability that evidence is sufficiently informative"
#
# Using Beta-Bernoulli with continuous noisy evidence:
#   - Each EvidenceItem contributes strength s_i ∈ (0,1)
#   - s_i = w_provider * provider_score + w_sim * sim
#   - Posterior: Beta(alpha, beta) where:
#     - alpha = alpha0 + Σ s_i
#     - beta = beta0 + Σ (1 - s_i)
#
# Uncertainty = normalized variance of posterior

# Evidence combination weights (policy priors, calibratable)
W_PROVIDER = 0.6  # Weight for provider_score
W_SIM = 0.4  # Weight for similarity score

# Prior hyperparameters (uninformative)
ALPHA0 = 1.0
BETA0 = 1.0

# Maximum variance for Beta(1,1)
VAR_MAX = 1.0 / 12.0


def _compute_evidence_strength(item: Any) -> float:
    """Compute evidence strength s_i ∈ (0.01, 0.99) from item scores."""
    provider = getattr(item, "provider_score", None)
    sim = getattr(item, "sim", None)
    provider = provider if provider is not None else 0.5
    sim = sim if sim is not None else 0.5

    s = W_PROVIDER * provider + W_SIM * sim
    return max(0.01, min(0.99, float(s)))


def _compute_update_weight(item: Any) -> float:
    """How much to trust this item's strength when updating the posterior."""
    has_provider = getattr(item, "provider_score", None) is not None
    has_sim = getattr(item, "sim", None) is not None
    if has_provider and has_sim:
        return 1.0
    if has_provider or has_sim:
        return 0.6
    return 0.2


def _collect_all_items(evidence_index: Any) -> list[Any]:
    """Collect evidence items from index (by_claim + global), de-duplicated by URL."""
    items: list[Any] = []
    seen: set[str] = set()

    def _push(it: Any) -> None:
        url = (getattr(it, "url", None) or "").strip()
        key = url or f"__no_url__:{id(it)}"
        if key in seen:
            return
        seen.add(key)
        items.append(it)

    for pack in getattr(evidence_index, "by_claim_id", {}).values():
        for it in getattr(pack, "items", ()) or ():
            _push(it)

    global_pack = getattr(evidence_index, "global_pack", None)
    if global_pack:
        for it in getattr(global_pack, "items", ()) or ():
            _push(it)

    return items


def compute_beta_uncertainty(evidence_index: Any) -> float:
    """Compute uncertainty as normalized Beta posterior variance."""
    items = _collect_all_items(evidence_index)

    if not items:
        return 1.0  # Maximum uncertainty with no evidence

    alpha = ALPHA0
    beta = BETA0

    for item in items:
        s = _compute_evidence_strength(item)
        w = _compute_update_weight(item)
        alpha += w * s
        beta += w * (1.0 - s)

    # Beta posterior variance
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    # Normalize to [0, 1]
    return var / VAR_MAX


def compute_delta_utility(uncertainty: float, base_utility: float = 0.05) -> float:
    """Compute expected utility improvement from running a step."""
    k = 0.15
    return base_utility + k * uncertainty


# =============================================================================
# FEATURE EXTRACTION FROM EVIDENCE INDEX
# =============================================================================

def extract_evidence_features(evidence_index: Any, claims: list[dict]) -> dict[str, float]:
    """
    Extracts summary features from the current evidence state for gating logic.
    """
    all_items = _collect_all_items(evidence_index)
    n_evidence = len(all_items)
    n_claims = len(claims)

    if n_evidence == 0:
        return {
            "n_evidence": 0.0,
            "n_claims": float(n_claims),
            "unlabeled_ratio": 1.0,
            "low_tier_ratio": 0.0,
            "log_claims": math.log(max(1, n_claims)),
        }

    unlabeled = 0
    low_tier = 0
    for it in all_items:
        # check stance
        stance = getattr(it, "stance", None)
        if stance is None:
            unlabeled += 1
        
        # check tier
        tier = getattr(it, "tier", "D")
        if tier in ("C", "D"):
            low_tier += 1

    return {
        "n_evidence": float(n_evidence),
        "n_claims": float(n_claims),
        "unlabeled_ratio": unlabeled / n_evidence,
        "low_tier_ratio": low_tier / n_evidence,
        "log_claims": math.log(max(1, n_claims)),
    }
