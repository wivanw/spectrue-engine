# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""deep_v2 tiered search: the general tier is decided after stance, on stance-annotated evidence. No network."""

from __future__ import annotations

import pytest

from spectrue_core.pipeline.contracts import EvidenceIndex, EvidencePackContract
from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.mode import DEEP_V2_MODE
from spectrue_core.pipeline.steps.retrieval.general_tier_search import GeneralTierSearchStep
from spectrue_core.use_cases.retrieval.tiered_search import (
    clusters_insufficient_after_stance, merge_by_claim, merge_index, merge_sources,
)

PLANS = [
    {"cluster_id": "k1", "search_queries": ["q1"], "claims": [{"id": "c1", "text": "VK reported a hit on a data center."}]},
    {"cluster_id": "k2", "search_queries": ["q2"], "claims": [{"id": "c2", "text": "A bridge was hit."}]},
]


def _src(cid, domain, quote=None, stance=None):
    return {"claim_id": cid, "url": f"https://{domain}/{cid}", "quote": quote, "stance": stance}


def test_quotes_and_stance_make_a_cluster_sufficient():
    sources = [_src("c1", "reuters.com", quote="VK said its data center was hit", stance="support"),
               _src("c2", "reuters.com")]                                 # c2: read, no quote, no stance
    assert set(clusters_insufficient_after_stance(PLANS, sources)) == {"k2"}


def test_merge_sources_and_by_claim_do_not_duplicate_claim_url_pairs():
    a = [_src("c1", "a.com"), _src("c2", "b.com")]
    b = [_src("c1", "a.com"), _src("c1", "c.com")]
    assert [s["url"] for s in merge_sources(a, b)] == ["https://a.com/c1", "https://b.com/c2", "https://c.com/c1"]
    assert len(merge_by_claim({"c1": a[:1]}, {"c1": b, "c3": [_src("c3", "d.com")]})["c1"]) == 2


def test_merge_index_appends_items_and_recomputes_missing():
    main = EvidenceIndex(by_claim_id={"c1": EvidencePackContract(items=("x",))}, missing_claims=("c2",))
    extra = EvidenceIndex(by_claim_id={"c1": EvidencePackContract(items=("y",)), "c2": EvidencePackContract(items=("z",))})
    out = merge_index(main, extra)
    assert out.by_claim_id["c1"].items == ("x", "y") and out.missing_claims == ()


class _Step:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    async def run(self, ctx):
        self.calls.append(ctx)
        return self.fn(ctx)


@pytest.mark.asyncio
async def test_step_runs_the_round_only_for_insufficient_clusters_and_merges():
    ctx = (PipelineContext(mode=DEEP_V2_MODE, claims=[c for p in PLANS for c in p["claims"]])
           .with_update(sources=[_src("c1", "reuters.com", quote="q", stance="support"), _src("c2", "reuters.com")])
           .set_extra("cluster_search_plans", PLANS).set_extra("stance_annotations", ["stale"])
           .set_extra("evidence_docs", {"https://reuters.com/c1": {}}))
    search = _Step(lambda c: c.set_extra("evidence_docs", {"https://other.org/c2": {}}))
    stance = _Step(lambda c: c.with_update(sources=[_src("c2", "other.org", quote="bridge hit", stance="support")]))
    out = await GeneralTierSearchStep(round_steps=(search, stance)).run(ctx)
    sub = search.calls[0]
    assert [p["cluster_id"] for p in sub.get_extra("cluster_search_plans")] == ["k2"]   # only the insufficient one
    assert sub.get_extra("stance_annotations") is None                                   # no stale stance reuse
    assert [s["url"] for s in out.sources] == ["https://reuters.com/c1", "https://reuters.com/c2", "https://other.org/c2"]
    assert set(out.get_extra("evidence_docs")) == {"https://reuters.com/c1", "https://other.org/c2"}


@pytest.mark.asyncio
async def test_step_is_a_no_op_when_every_cluster_is_sufficient():
    ctx = (PipelineContext(mode=DEEP_V2_MODE, claims=[])
           .with_update(sources=[_src("c1", "reuters.com", quote="q", stance="support"),
                                 _src("c2", "reuters.com", quote="q", stance="support")])
           .set_extra("cluster_search_plans", PLANS))
    search = _Step(lambda c: c)
    out = await GeneralTierSearchStep(round_steps=(search,)).run(ctx)
    assert not search.calls and out is ctx
