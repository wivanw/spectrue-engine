# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Claim Metadata Parsing

This module isolates parsing/validation of Claim metadata (M80) from the
ClaimExtractionSkill implementation, so the main skill code reads as a pipeline.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    ClaimRole,
    EvidenceChannel,
    MetadataConfidence,
    RetrievalPolicy,
    SearchLocalePlan,
    VerificationTarget,
)


def default_channels(
    *,
    harm_potential: int,
    verification_target: VerificationTarget,
) -> list[EvidenceChannel]:
    """
    Determine default channels based on harm and target.

    High harm → authoritative only
    No verification target → empty (no search)
    Normal → authoritative + reputable + local
    """
    if verification_target == VerificationTarget.NONE:
        return []  # No search needed

    if harm_potential >= 4:
        return [EvidenceChannel.AUTHORITATIVE]

    return [
        EvidenceChannel.AUTHORITATIVE,
        EvidenceChannel.REPUTABLE_NEWS,
        EvidenceChannel.LOCAL_MEDIA,
    ]


def parse_claim_metadata(
    rc: dict[str, Any],
    *,
    lang: str,
    harm_potential: int,
    claim_category: str,
    satire_likelihood: float,
) -> ClaimMetadata:
    """
    Parse Claim metadata fields from raw LLM output.

    Returns ClaimMetadata with metadata_confidence=LOW if many fields missing.
    """
    missing_count = 0

    # 1) verification_target
    vt_raw = rc.get("verification_target", "")
    if vt_raw:
        try:
            verification_target = VerificationTarget(str(vt_raw).lower())
        except ValueError:
            verification_target = VerificationTarget.REALITY
            missing_count += 1
    else:
        if claim_category in {"SATIRE", "OPINION", "HYPERBOLIC"} or satire_likelihood >= 0.7:
            verification_target = VerificationTarget.NONE
        else:
            verification_target = VerificationTarget.REALITY
        missing_count += 1

    # 2) claim_role
    cr_raw = rc.get("claim_role", "")
    if cr_raw:
        try:
            claim_role = ClaimRole(str(cr_raw).lower())
        except ValueError:
            claim_role = ClaimRole.CORE
            missing_count += 1
    else:
        claim_role = ClaimRole.CONTEXT if verification_target == VerificationTarget.NONE else ClaimRole.CORE
        missing_count += 1

    # 3) search_locale_plan
    slp_raw = rc.get("search_locale_plan", {})
    if isinstance(slp_raw, dict) and slp_raw:
        primary = slp_raw.get("primary", lang) or lang
        fallback = slp_raw.get("fallback", ["en"])
        if not isinstance(fallback, list):
            fallback = [fallback] if fallback else ["en"]
        search_locale_plan = SearchLocalePlan(primary=str(primary), fallback=[str(x) for x in fallback])
    else:
        search_locale_plan = SearchLocalePlan(primary=lang, fallback=["en"])
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
        channels: list[EvidenceChannel] = []
        for c in (channels_raw or []):
            try:
                cc = str(c)
                if cc == "low_reliability":
                    cc = EvidenceChannel.LOW_RELIABILITY.value
                channels.append(EvidenceChannel(cc))
            except ValueError:
                continue

        use_policy_raw = rp_raw.get("use_policy_by_channel")
        if use_policy_raw is None:
            use_policy_raw = rp_raw.get("use_policy", {})

        retrieval_policy = RetrievalPolicy(
            channels_allowed=channels if channels else default_channels(harm_potential=harm_potential, verification_target=verification_target),
            use_policy=use_policy_raw if isinstance(use_policy_raw, dict) and use_policy_raw else {},
        )
    else:
        retrieval_policy = RetrievalPolicy(
            channels_allowed=default_channels(harm_potential=harm_potential, verification_target=verification_target),
        )
        missing_count += 1

    # 5) metadata_confidence
    mc_raw = rc.get("metadata_confidence", "")
    if mc_raw:
        try:
            metadata_confidence = MetadataConfidence(str(mc_raw).lower())
        except ValueError:
            metadata_confidence = MetadataConfidence.MEDIUM
    else:
        if missing_count >= 3:
            metadata_confidence = MetadataConfidence.LOW
        elif missing_count >= 1:
            metadata_confidence = MetadataConfidence.MEDIUM
        else:
            metadata_confidence = MetadataConfidence.HIGH

    check_worthiness = float(rc.get("check_worthiness", 0.5))
    check_worthiness = max(0.0, min(1.0, check_worthiness))

    return ClaimMetadata(
        verification_target=verification_target,
        claim_role=claim_role,
        check_worthiness=check_worthiness,
        search_locale_plan=search_locale_plan,
        time_signals=time_signals,
        locale_signals=locale_signals,
        time_sensitive=time_sensitive,
        retrieval_policy=retrieval_policy,
        metadata_confidence=metadata_confidence,
    )
