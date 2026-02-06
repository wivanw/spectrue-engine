"""Claim metadata normalization and parsing helpers."""

from __future__ import annotations

from typing import Any

from spectrue_core.schema import ClaimDomain, ClaimType


TOPIC_GROUPS = [
    "Politics",
    "Economy",
    "War",
    "Science",
    "Technology",
    "Health",
    "Environment",
    "Society",
    "Sports",
    "Culture",
    "Other",
]

ARTICLE_INTENTS = ["news", "evergreen", "official", "opinion", "prediction", "unknown", "other"]

ALLOWED_CLAIM_CATEGORIES = {"FACTUAL", "SATIRE", "OPINION", "HYPERBOLIC"}

DOMAIN_MAPPING = {
    "Politics": ClaimDomain.POLITICS,
    "Economy": ClaimDomain.FINANCE,
    "War": ClaimDomain.NEWS,
    "Science": ClaimDomain.SCIENCE,
    "Technology": ClaimDomain.TECHNOLOGY,
    "Health": ClaimDomain.HEALTH,
    "Environment": ClaimDomain.SCIENCE,
    "Society": ClaimDomain.NEWS,
    "Sports": ClaimDomain.SPORTS,
    "Culture": ClaimDomain.ENTERTAINMENT,
    "Other": ClaimDomain.OTHER,
}

CLAIM_TYPE_MAPPING = {
    "core": ClaimType.EVENT,
    "numeric": ClaimType.NUMERIC,
    "timeline": ClaimType.TIMELINE,
    "attribution": ClaimType.ATTRIBUTION,
    "sidefact": ClaimType.OTHER,
    "atomic": ClaimType.OTHER,
    "causal": ClaimType.OTHER,
    "comparative": ClaimType.COMPARISON,
    "policy_plan": ClaimType.POLICY,
    "definition": ClaimType.DEFINITION,
    "future": ClaimType.TIMELINE,
    "existence": ClaimType.OTHER,
}


def clamp_float(value: Any, *, default: float, lo: float, hi: float) -> float:
    try:
        f = float(value)
    except Exception:
        f = float(default)
    return max(lo, min(hi, f))


def clamp_int(value: Any, *, default: int, lo: int, hi: int) -> int:
    try:
        i = int(value)
    except Exception:
        i = int(default)
    return max(lo, min(hi, i))


def normalize_topic_group(topic: str | None, *, default: str = "Other") -> str:
    t = (topic or default) or default
    return t if t in TOPIC_GROUPS else default


def normalize_article_intent(intent: str | None, *, default: str = "news") -> str:
    v = (intent or default) or default
    return v if v in ARTICLE_INTENTS else default


def normalize_claim_category(category: str | None, *, default: str = "FACTUAL") -> str:
    v = (category or default) or default
    return v if v in ALLOWED_CLAIM_CATEGORIES else default


def normalize_channel_token(token: str) -> str:
    return token.strip().lower().replace("-", "_").replace(" ", "_")


def default_channels_values(*, harm_potential: int, verification_target: str) -> list[str]:
    """Determine default channels based on harm and target (string values)."""
    if verification_target == "none":
        return []

    if harm_potential >= 4:
        return ["authoritative"]

    return ["authoritative", "reputable_news", "local_media"]


def parse_claim_metadata_fields(
    rc: dict[str, Any],
    *,
    lang: str,
    harm_potential: int,
    claim_category: str,
    satire_likelihood: float,
) -> dict[str, Any]:
    """Parse Claim metadata fields from raw LLM output into primitives."""
    missing_count = 0

    # 1) verification_target
    vt_raw = rc.get("verification_target", "")
    if vt_raw:
        verification_target = str(vt_raw).lower()
        if verification_target not in {"reality", "existence", "attribution", "none"}:
            verification_target = "reality"
            missing_count += 1
    else:
        if claim_category in {"SATIRE", "OPINION", "HYPERBOLIC"} or satire_likelihood >= 0.7:
            verification_target = "none"
        else:
            verification_target = "reality"
        missing_count += 1

    # 2) claim_role
    cr_raw = rc.get("claim_role", "")
    if cr_raw:
        claim_role = str(cr_raw).lower()
        if claim_role not in {"core", "context", "quote", "statistic"}:
            claim_role = "core"
            missing_count += 1
    else:
        claim_role = "context" if verification_target == "none" else "core"
        missing_count += 1

    # 3) search_locale_plan
    slp_raw = rc.get("search_locale_plan", {})
    if isinstance(slp_raw, dict) and slp_raw:
        primary = slp_raw.get("primary", lang) or lang
        fallback = slp_raw.get("fallback", ["en"])
        if not isinstance(fallback, list):
            fallback = [fallback] if fallback else ["en"]
        search_locale_plan = {"primary": str(primary), "fallback": [str(x) for x in fallback]}
    else:
        search_locale_plan = {"primary": lang, "fallback": ["en"]}
        missing_count += 1

    # 3.5) temporal/locale signals
    time_signals_raw = rc.get("time_signals") or rc.get("temporal_signals") or []
    if isinstance(time_signals_raw, dict):
        time_signals_raw = [time_signals_raw]
    time_signals = [s for s in time_signals_raw if isinstance(s, dict)]

    locale_signals_raw = rc.get("locale_signals") or []
    if isinstance(locale_signals_raw, dict):
        locale_signals_raw = [locale_signals_raw]
    locale_signals = [s for s in locale_signals_raw if isinstance(s, dict)]

    time_sensitive_raw = rc.get("time_sensitive")
    if time_sensitive_raw is None:
        time_sensitive_raw = rc.get("is_time_sensitive")
    time_sensitive = bool(time_sensitive_raw) or bool(time_signals)

    # 4) retrieval_policy
    rp_raw = rc.get("retrieval_policy", {})
    if isinstance(rp_raw, dict) and rp_raw:
        channels_raw = rp_raw.get("channels_allowed", [])
        channels: list[str] = []
        for c in (channels_raw or []):
            cc = normalize_channel_token(str(c))
            if cc == "low_reliability":
                cc = "low_reliability"
            if cc:
                channels.append(cc)

        use_policy_raw = rp_raw.get("use_policy_by_channel")
        if use_policy_raw is None:
            use_policy_raw = rp_raw.get("use_policy", {})

        retrieval_policy = {
            "channels_allowed": channels if channels else default_channels_values(
                harm_potential=harm_potential,
                verification_target=verification_target,
            ),
            "use_policy": use_policy_raw if isinstance(use_policy_raw, dict) and use_policy_raw else {},
        }
    else:
        retrieval_policy = {
            "channels_allowed": default_channels_values(
                harm_potential=harm_potential,
                verification_target=verification_target,
            ),
        }
        missing_count += 1

    # 5) metadata_confidence
    mc_raw = rc.get("metadata_confidence", "")
    if mc_raw:
        metadata_confidence = str(mc_raw).lower()
        if metadata_confidence not in {"low", "medium", "high"}:
            metadata_confidence = "medium"
    else:
        if missing_count >= 3:
            metadata_confidence = "low"
        elif missing_count >= 1:
            metadata_confidence = "medium"
        else:
            metadata_confidence = "high"

    check_worthiness = float(rc.get("check_worthiness", 0.5))
    check_worthiness = max(0.0, min(1.0, check_worthiness))

    from spectrue_core.utils.topic_tags import build_topic_tags

    topic_tags = build_topic_tags(
        topic_group=rc.get("topic_group"),
        topic_key=rc.get("topic_key"),
        subject_entities=rc.get("subject_entities") or [],
        retrieval_seed_terms=rc.get("retrieval_seed_terms") or [],
        verification_target=verification_target,
        claim_role=claim_role,
        locale_signals=locale_signals,
        time_signals=time_signals,
    )

    return {
        "verification_target": verification_target,
        "claim_role": claim_role,
        "check_worthiness": check_worthiness,
        "search_locale_plan": search_locale_plan,
        "time_signals": time_signals,
        "locale_signals": locale_signals,
        "time_sensitive": time_sensitive,
        "retrieval_policy": retrieval_policy,
        "metadata_confidence": metadata_confidence,
        "topic_tags": topic_tags,
    }
