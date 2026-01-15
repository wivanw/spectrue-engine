# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass

from spectrue_core.pipeline.core import PipelineContext, Step
from spectrue_core.pipeline.mode import AnalysisMode
from spectrue_core.utils.trace import Trace
from spectrue_core.verification.evidence.dedup_fingerprints import (
    evidence_text_payload,
    normalize_publisher,
    normalize_text_for_hash,
    sha256_hex,
    simhash64,
    simhash_bucket_id,
)


@dataclass
class EvidenceDedupStep(Step):
    """
    Compute exact-dup and near-dup fingerprints for EvidenceItems.
    - publisher_id: normalized domain
    - content_hash: sha256(normalized payload)
    - similar_cluster_id: simhash bucket id
    """

    @property
    def name(self) -> str:
        return "evidence_dedup"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.mode.api_analysis_mode != AnalysisMode.DEEP_V2:
            return ctx

        sources = ctx.sources or []
        if not sources:
            return ctx

        updated = 0
        exact_groups = 0
        sim_groups = 0
        seen_hash = set()
        seen_sim = set()

        for s in sources:
            if not isinstance(s, dict):
                continue
            domain = str(s.get("domain") or "")
            pub = normalize_publisher(domain)
            s["publisher_id"] = pub

            payload = evidence_text_payload(s)
            norm = normalize_text_for_hash(payload)
            ch = sha256_hex(norm) if norm else ""
            s["content_hash"] = ch

            sh = simhash64(payload) if payload else 0
            scid = simhash_bucket_id(sh, prefix_bits=16) if payload else ""
            s["similar_cluster_id"] = scid

            if ch and ch not in seen_hash:
                seen_hash.add(ch)
                exact_groups += 1
            if scid and scid not in seen_sim:
                seen_sim.add(scid)
                sim_groups += 1

            updated += 1

        Trace.event(
            "evidence_dedup.completed",
            {"items": updated, "exact_groups": exact_groups, "similar_groups": sim_groups},
        )
        return ctx
