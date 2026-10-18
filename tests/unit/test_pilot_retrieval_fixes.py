# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fixes from the 2026-10-18 fact-checker pilot (trace 2026-10-18_17-39-01_aaaf35, Nigeria urban population)."""

from __future__ import annotations

import pytest

from spectrue_core.domain.verification.verdict.model import AnalysisMode, EvidenceCollection, EvidenceFlowInput
from spectrue_core.pipeline.evidence_flow import _deduplicated
from spectrue_core.pipeline.retrieval.cegs_mvp import (
    EvidenceItem, EvidencePool, EvidenceSourceMeta, build_doc_query_plan, match_claim_to_pool,
)
from spectrue_core.use_cases.verification.pipeline_queries import preplanned_global_queries

CLAIM = {
    "id": "c1",
    "text": "Nigeria’s urban population at independence was approximately 7 million.",
    "subject_entities": ["Nigeria"],
    "retrieval_seed_terms": ["Nigeria", "urban", "population", "1960", "independence", "7 million"],
    "search_queries": ["Nigeria urban population 1960", "Nigeria independence 1960 urban population 7 million"],
}


def _item(url: str, title: str, text: str, score: float) -> EvidenceItem:
    meta = EvidenceSourceMeta(url=url, title=title, snippet="", score=score)
    return EvidenceItem(url=url, extracted_text=text, content_hash=url, source_meta=meta)


def test_pool_match_prefers_pages_about_the_claim_over_pages_about_the_subject():
    pool = EvidencePool()
    pool.add_items([
        _item("https://a/nigeria", "Nigeria", "Nigeria is a country in West Africa.", 0.90),
        _item("https://b/kids", "Nigeria facts", "Nigeria became a country.", 0.89),
        _item("https://c/wb", "Nigeria data", "Nigeria country page.", 0.88),
        _item("https://d/demographics", "Demographics of Nigeria",
              "In 1960, at independence, Nigeria's urban population was about 7 million.", 0.80),
    ])
    urls = [i.url for i in match_claim_to_pool(CLAIM, pool).matched_items]
    assert urls[0] == "https://d/demographics"


def test_doc_plan_never_repeats_a_word():
    for q in build_doc_query_plan([CLAIM], anchors=[]):
        words = [w.lower() for w in q.split()]
        assert len(words) == len(set(words)), q


def test_standard_global_queries_come_from_the_claims_own_queries():
    assert preplanned_global_queries([CLAIM], max_queries=1) == ["Nigeria urban population 1960"]
    other = {"search_queries": ["Lagos population 1960"]}
    assert preplanned_global_queries([CLAIM, other], max_queries=2) == [
        "Nigeria urban population 1960", "Lagos population 1960"]   # round-robin across claims
    assert preplanned_global_queries([{"text": "no queries"}], max_queries=3) == []


def test_judge_pack_holds_no_exact_duplicates():
    src = {"url": "https://a/nigeria", "domain": "a", "quote": "Nigeria became independent in 1960.",
           "claim_id": "c1", "stance": "context"}
    collection = EvidenceCollection(pack={"context_sources": [src, dict(src)]}, claims=[CLAIM], sources=[src, dict(src)],
                                    claim_text_map={"c1": CLAIM["text"]}, anchor_claim=CLAIM, anchor_claim_id="c1",
                                    time_windows={}, current_cost=0.0)
    inp = EvidenceFlowInput(fact=CLAIM["text"], original_fact=CLAIM["text"], lang="en", content_lang="en",
                            analysis_mode=AnalysisMode.GENERAL, progress_callback=None)
    out = _deduplicated(collection, inp)
    assert len(out.sources) == 1
    urls = [s.get("url") for s in (out.pack.get("scored_sources") or []) + (out.pack.get("context_sources") or [])]
    assert urls.count("https://a/nigeria") == 1


def test_same_passage_for_two_claims_is_not_a_duplicate():
    """Pilot 2026-10-18 (Bon Jovi): dedup across claims dropped all of c1's evidence."""
    from spectrue_core.use_cases.evidence.dedup import apply_dedup
    a = {"url": "https://x/rally", "domain": "x", "quote": "Bon Jovi performed at the drive-in rally.", "claim_id": "c1"}
    b = {**a, "claim_id": "c2"}
    kept, stats = apply_dedup(sources=[a, b, dict(a)], filter_duplicates=True)
    assert sorted(s["claim_id"] for s in kept) == ["c1", "c2"] and stats["removed"] == 1


# --- Bugs found by the 2026-10-18 classification/heuristics survey ---

def test_oracle_stance_follows_the_fact_checkers_verdict_not_relevance():
    from spectrue_core.pipeline.steps.oracle_flow import OracleFlowStep
    from spectrue_core.use_cases.verification.pipeline_oracle import oracle_stance, oracle_verified_score
    refuted = {"status": "REFUTED", "relevance_score": 0.85, "url": "u"}
    assert OracleFlowStep._create_evidence_source(refuted)["stance"] == "REFUTE"   # was SUPPORT (relevance > 0.7)
    assert oracle_stance("CONFIRMED") == "SUPPORT" and oracle_stance("MIXED") == "CONTEXT"
    assert oracle_verified_score({"status": "REFUTED", "relevance_score": 0.95}) == pytest.approx(0.05)
    assert oracle_verified_score({"status": "MIXED", "relevance_score": 0.95, "verified_score": 0.4}) == 0.4  # was 0.95
    assert oracle_verified_score({"status": "MIXED", "relevance_score": 0.95}) == -1.0


def test_archive_pdf_with_current_year_counts_as_news():
    import datetime
    from spectrue_core.tools.search_scoring import rank_and_filter
    year = datetime.datetime.now().year
    res = rank_and_filter("budget report", [{"url": "https://gov.example/r.pdf", "title": f"Budget report {year}",
                                             "content": "budget report figures", "score": 0.9}])
    assert [r["url"] for r in res] == ["https://gov.example/r.pdf"]   # the escaped regex never matched a year


def test_freshness_penalty_only_for_time_sensitive_claims():
    """Standard mode called the freshness check without time_sensitive (default True): every claim was penalised."""
    from unittest.mock import patch
    from spectrue_core.use_cases.verification.scoring.evidence_verdict_processing import process_single_claim_verdict
    seen = []

    def fake(claim_id, items, current_year=None, is_time_sensitive=True):
        seen.append(is_time_sensitive)
        return 0.0
    with patch("spectrue_core.use_cases.verification.scoring.freshness_signal.calculate_freshness_adjustment", fake):
        for ts in (False, True, None):
            process_single_claim_verdict(cv={"claim_id": "c1", "verdict_score": 0.9}, claim_id="c1",
                                         claim_obj={"id": "c1", "time_sensitive": ts}, temporality=None,
                                         has_direct_evidence=True, pack={"items": []}, explainability=0.8)
    assert seen == [False, True, False]


def test_pool_match_breaks_equal_coverage_by_search_relevance():
    """Masks claim: 12 pages tied at the same overlap; the first three by insertion order were taken."""
    pool = EvidencePool()
    pool.add_items([_item(f"https://p/{i}", "Nigeria urban population 1960 independence",
                          "Nigeria urban population 1960 independence 7 million.", score) for i, score in
                    enumerate([0.2, 0.3, 0.95, 0.4, 0.9, 0.85])])
    urls = [i.url for i in match_claim_to_pool(CLAIM, pool).matched_items]
    assert urls == ["https://p/2", "https://p/4", "https://p/5"]


def test_claim_frame_keeps_evidence_whose_text_is_in_content_excerpt():
    """Deep trace 2026-10-18_20-22-24: the VK article was dropped as an 'empty snippet'."""
    from spectrue_core.pipeline.claims.claim_frame_builder import convert_evidence_items
    ev = [{"url": "https://24tv.ua/vk", "content_excerpt": "The VK data center in St Petersburg was hit by a drone overnight, the company said in a statement."},
          {"url": "https://x/empty"}]
    items = convert_evidence_items("c4", ev)
    assert [i.url for i in items] == ["https://24tv.ua/vk"]


def test_claim_type_comes_from_predicate_type_not_regex():
    """M147 T032: a short claim is not a sidefact; a year in the text is not a timeline."""
    from spectrue_core.domain.claims.graph.nodes import _infer_claim_type
    assert _infer_claim_type({"claim_role": "core", "text": "VK was hit.", "predicate_type": "event"}) == "timeline"
    assert _infer_claim_type({"claim_role": "core", "text": "Short claim", "predicate_type": "property"}) == "core"
    assert _infer_claim_type({"claim_role": "core", "text": "In 2018 it said 5%", "predicate_type": "quote"}) == "attribution"
    assert _infer_claim_type({"claim_role": "core", "text": "x", "predicate_type": "measurement"}) == "numeric"
    assert _infer_claim_type({"claim_role": "background", "text": "x", "predicate_type": "measurement"}) == "sidefact"
