from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from spectrue_core.billing import pricing, token_estimator
from spectrue_core.billing.config_loader import load_pricing_config
from spectrue_core.llm.model_registry import MODEL_MID, MODEL_NANO, MODEL_PRO


RequiredCapability = Literal["cheap", "mid", "high"]


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


def _estimate_credits_for_model(*, model: str, prompt_chars: int, out_tokens: int) -> float:
    """
    Deterministic cost estimate in *credits* (not USD).
    Uses pricing config registry; if missing, pricing module provides a safe fallback mapping.
    """
    cfg = load_pricing_config()
    mp = pricing.get_model_pricing(cfg, model)
    in_tokens = token_estimator.estimate_tokens_from_chars(prompt_chars)
    return pricing.estimate_cost_credits(mp, input_tokens=in_tokens, output_tokens=out_tokens)


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
      - gpt-5-nano: cheapest, only for truly simple/low-risk claims with clean evidence
      - deepseek-chat: mid tier for medium complexity, BUT with explicit failure fallback to gpt-5.2
      - gpt-5.2: for high importance / high harm / conflict / high ambiguity / low coverage

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
    # Central / important / harmful / conflicting => gpt-5.2 directly
    if importance >= 0.75 or harm_norm >= 0.75 or has_conflict or ambiguity >= 0.70:
        required_cap: RequiredCapability = "high"
    elif difficulty <= 0.35 and risk <= 0.35 and n_domains >= 2 and hi_tier_ratio >= 0.30:
        required_cap = "cheap"
    else:
        required_cap = "mid"

    allowed: list[str]
    if required_cap == "high":
        allowed = [MODEL_PRO]
    elif required_cap == "cheap":
        allowed = [MODEL_NANO, MODEL_MID, MODEL_PRO]
    else:
        allowed = [MODEL_MID, MODEL_PRO]

    # ---- price-aware selection: choose cheapest expected credits among allowed ----
    est = {
        MODEL_NANO: _estimate_credits_for_model(model=MODEL_NANO, prompt_chars=prompt_chars, out_tokens=out_tokens_estimate),
        MODEL_MID: _estimate_credits_for_model(model=MODEL_MID, prompt_chars=prompt_chars, out_tokens=out_tokens_estimate),
        MODEL_PRO: _estimate_credits_for_model(model=MODEL_PRO, prompt_chars=prompt_chars, out_tokens=out_tokens_estimate),
    }

    # expected credits include Deepseek failure probability *fallback cost*
    expected = dict(est)
    if MODEL_MID in expected:
        expected[MODEL_MID] = (1.0 - deepseek_fail_prob) * est[MODEL_MID] + deepseek_fail_prob * est[MODEL_PRO]

    chosen = min(allowed, key=lambda m: expected[m])

    if chosen == MODEL_NANO:
        reason = "cheap_ok_min_expected_cost"
    elif chosen == MODEL_MID:
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
        fallback_model=MODEL_PRO,
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
