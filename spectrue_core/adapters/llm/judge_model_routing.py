from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal

from spectrue_core.billing import pricing, token_estimator
from spectrue_core.billing.config_loader import load_pricing_config
from spectrue_core.billing.types import CreditPricingPolicy
from spectrue_core.llm.model_registry import ModelID

logger = logging.getLogger(__name__)


RequiredCapability = Literal["cheap", "mid", "high"]

# Cost assigned to a model that has no entry in the pricing policy. Large enough
# that such a model is never *preferred*, but finite so that min() stays well
# defined (and traces stay JSON-serializable) if it is the only allowed option.
UNPRICED_MODEL_CREDITS = 1e9


@dataclass(frozen=True)
class JudgeModelDecision:
    model: str
    fallback_model: str
    required_capability: RequiredCapability

    # continuous signals (0..1)
    difficulty: float
    risk: float
    complexity: float
    ambiguity: float

    # cost (credits) – deterministic estimate
    est_credits: dict[str, float]
    expected_credits: dict[str, float]

    # debug features for traceability
    features: dict[str, Any]
    reason: str

    def to_trace(self) -> dict[str, Any]:
        return asdict(self)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _norm_int(x: int, hi: int) -> float:
    if hi <= 0:
        return 0.0
    return _clamp01(float(x) / float(hi))


@lru_cache(maxsize=1)
def _pricing_policy() -> CreditPricingPolicy:
    """Pricing policy, loaded once.

    ``load_pricing_config`` reads and parses a JSON file, and routing runs once
    per claim, so this is cached. Pricing is process-static; tests that need a
    different policy can call ``_pricing_policy.cache_clear()``.
    """
    return load_pricing_config()


def _estimate_credits_for_model(*, model: str, prompt_chars: int, out_tokens: int) -> float:
    """
    Deterministic cost estimate in *credits* (not USD).

    NOTE: reasoning tokens are not modelled — they are unpredictable before the
    call. For reasoning models billed per reasoning token (the PRO tier) this
    understates the true cost, so routing is, if anything, biased *towards* the
    expensive tier rather than away from it.
    """
    policy = _pricing_policy()
    price = policy.get_model_price(model)
    if price is None:
        logger.warning(
            "[judge_routing] No pricing entry for model %r; treating as unaffordable", model
        )
        return UNPRICED_MODEL_CREDITS

    in_tokens = token_estimator.estimate_tokens_from_chars(prompt_chars)
    credits = pricing.llm_usage_to_credits(
        price=price,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        reasoning_tokens=None,
        policy=policy,
    )
    # Callers do plain float arithmetic on these (expected-cost blending), and
    # JudgeModelDecision declares dict[str, float].
    return float(credits)


def select_judge_model(
    *,
    claim: Any,
    evidence_items: list[Any],
    prompt_chars: int,
    out_tokens_estimate: int = 380,
    deepseek_fail_prob: float = 0.18,
) -> JudgeModelDecision:
    """
    Price-aware, deterministic routing for judge model:
      - gpt-5.6-luna (NANO): cheapest, only for truly simple/low-risk claims with clean evidence
      - deepseek-v4-flash (MID): mid tier for medium complexity, BUT with explicit failure fallback to PRO
      - gpt-5.6-sol (PRO): for high importance / high harm / conflict / high ambiguity / low coverage

    Inputs:
      claim: expects fields like importance, check_worthiness, claim_role, harm_potential (optional)
      evidence_items: list with at least source_reliability_hint / domain / stance / has_quote (if present)
      prompt_chars: the constructed judge prompt size (chars)
    """
    # ---- claim signals (robust to missing fields) ----
    importance = float(getattr(claim, "importance", 0.5) or 0.5)
    worthiness = float(getattr(claim, "check_worthiness", 0.5) or 0.5)
    role = str(getattr(claim, "claim_role", "") or "")
    harm = getattr(claim, "harm_potential", None)
    if harm is None:
        # fallback: derive low harm if not present
        harm_norm = 0.2 if importance < 0.7 else 0.35
    else:
        harm_norm = _clamp01(float(harm) / 5.0)

    # ---- evidence signals ----
    n_items = len(evidence_items)
    domains: set[str] = set()
    n_quotes = 0
    st_support = 0
    st_refute = 0
    st_context = 0
    st_unknown = 0
    hi_tier = 0

    for it in evidence_items:
        d = getattr(it, "domain", None) or getattr(it, "source_domain", None)
        if d:
            domains.add(str(d).lower())

        if getattr(it, "has_quote", False):
            n_quotes += 1

        stance = getattr(it, "stance", None) or getattr(it, "stance_label", None)
        match str(stance or "").lower():
            case "support" | "supported":
                st_support += 1
            case "refute" | "refuted":
                st_refute += 1
            case "context" | "mention" | "neutral":
                st_context += 1
            case _:
                st_unknown += 1

        hint = getattr(it, "source_reliability_hint", None) or getattr(it, "tier", None)
        if hint in ("A", "B", "authoritative", "reputable_news"):
            hi_tier += 1

    n_domains = len(domains)
    unknown_ratio = (float(st_unknown) / float(n_items)) if n_items else 1.0
    has_conflict = (st_support > 0 and st_refute > 0)
    hi_tier_ratio = (float(hi_tier) / float(n_items)) if n_items else 0.0

    # ---- difficulty decomposition ----
    # complexity: more evidence items + more domains + more quotes => heavier prompt + more synthesis
    complexity = _clamp01(
        0.45 * _norm_int(n_items, 18)
        + 0.35 * _norm_int(n_domains, 6)
        + 0.20 * _norm_int(n_quotes, 8)
    )
    # ambiguity: stance unknown + conflict + low high-tier ratio
    ambiguity = _clamp01(
        0.55 * unknown_ratio
        + 0.30 * (1.0 if has_conflict else 0.0)
        + 0.15 * (1.0 - hi_tier_ratio)
    )
    # risk: harm + importance + worthiness
    risk = _clamp01(0.50 * harm_norm + 0.35 * importance + 0.15 * worthiness)

    difficulty = _clamp01(0.40 * complexity + 0.35 * ambiguity + 0.25 * risk)

    # ---- hard gates (quality constraints) ----
    # Central / important / harmful / conflicting => PRO tier directly
    if importance >= 0.75 or harm_norm >= 0.75 or has_conflict or ambiguity >= 0.70:
        required_cap: RequiredCapability = "high"
    elif difficulty <= 0.35 and risk <= 0.35 and n_domains >= 2 and hi_tier_ratio >= 0.30:
        required_cap = "cheap"
    else:
        required_cap = "mid"

    allowed: list[str]
    if required_cap == "high":
        allowed = [ModelID.PRO]
    elif required_cap == "cheap":
        allowed = [ModelID.NANO, ModelID.MID, ModelID.PRO]
    else:
        allowed = [ModelID.MID, ModelID.PRO]

    # ---- price-aware selection: choose cheapest expected credits among allowed ----
    est = {
        ModelID.NANO: _estimate_credits_for_model(model=ModelID.NANO, prompt_chars=prompt_chars, out_tokens=out_tokens_estimate),
        ModelID.MID: _estimate_credits_for_model(model=ModelID.MID, prompt_chars=prompt_chars, out_tokens=out_tokens_estimate),
        ModelID.PRO: _estimate_credits_for_model(model=ModelID.PRO, prompt_chars=prompt_chars, out_tokens=out_tokens_estimate),
    }

    # expected credits include Deepseek failure probability *fallback cost*
    expected = dict(est)
    if ModelID.MID in expected:
        expected[ModelID.MID] = (1.0 - deepseek_fail_prob) * est[ModelID.MID] + deepseek_fail_prob * est[ModelID.PRO]

    chosen = min(allowed, key=lambda m: expected[m])

    if chosen == ModelID.NANO:
        reason = "cheap_ok_min_expected_cost"
    elif chosen == ModelID.MID:
        reason = "mid_min_expected_cost_with_fallback"
    else:
        reason = "high_required_or_cheapest_allowed"

    features: dict[str, Any] = {
        "importance": importance,
        "worthiness": worthiness,
        "harm_norm": harm_norm,
        "role": role,
        "n_items": n_items,
        "n_domains": n_domains,
        "n_quotes": n_quotes,
        "support": st_support,
        "refute": st_refute,
        "context": st_context,
        "unknown": st_unknown,
        "has_conflict": has_conflict,
        "unknown_ratio": unknown_ratio,
        "hi_tier_ratio": hi_tier_ratio,
        "allowed": allowed,
    }

    return JudgeModelDecision(
        model=chosen,
        fallback_model=ModelID.PRO,
        required_capability=required_cap,
        difficulty=difficulty,
        risk=risk,
        complexity=complexity,
        ambiguity=ambiguity,
        est_credits=est,
        expected_credits=expected,
        features=features,
        reason=reason,
    )
