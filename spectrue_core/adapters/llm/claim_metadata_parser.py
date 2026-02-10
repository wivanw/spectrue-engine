# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Claim Metadata Parsing

This module isolates parsing/validation of Claim metadata for claim orchestration from the
ClaimExtractionSkill implementation, so the main skill code reads as a pipeline.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.domain.claims.metadata import (
    default_channels_values,
    normalize_channel_token,
    parse_claim_metadata_fields,
)
from spectrue_core.domain.claims.model import (
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
    values = default_channels_values(
        harm_potential=harm_potential,
        verification_target=verification_target.value,
    )
    out: list[EvidenceChannel] = []
    for v in values:
        try:
            out.append(EvidenceChannel(v))
        except ValueError:
            continue
    return out


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
    fields = parse_claim_metadata_fields(
        rc,
        lang=lang,
        harm_potential=harm_potential,
        claim_category=claim_category,
        satire_likelihood=satire_likelihood,
    )

    verification_target = VerificationTarget(fields["verification_target"])
    claim_role = ClaimRole(fields["claim_role"])
    search_locale_plan = SearchLocalePlan(**fields["search_locale_plan"])

    channels: list[EvidenceChannel] = []
    for c in fields["retrieval_policy"].get("channels_allowed", []):
        try:
            cc = normalize_channel_token(str(c))
            channels.append(EvidenceChannel(cc))
        except ValueError:
            continue

    if not channels:
        channels = default_channels(
            harm_potential=harm_potential,
            verification_target=verification_target,
        )

    use_policy = fields["retrieval_policy"].get("use_policy", {})
    retrieval_policy = RetrievalPolicy(
        channels_allowed=channels,
        use_policy=use_policy if isinstance(use_policy, dict) else {},
    )

    metadata_confidence = MetadataConfidence(fields["metadata_confidence"])

    return ClaimMetadata(
        verification_target=verification_target,
        claim_role=claim_role,
        check_worthiness=fields["check_worthiness"],
        search_locale_plan=search_locale_plan,
        time_signals=fields["time_signals"],
        locale_signals=fields["locale_signals"],
        time_sensitive=fields["time_sensitive"],
        retrieval_policy=retrieval_policy,
        metadata_confidence=metadata_confidence,
        topic_tags=fields["topic_tags"],
    )
