# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""deep_v2 cluster representatives cover every verifiable claim's subject entities."""

from __future__ import annotations

from spectrue_core.domain.claims.clustering import build_claim_clusters, select_representatives
from spectrue_core.domain.claims.graph.types import GraphResult

# Trace 2026-10-18_15-35-32: MST edges c4-c3 0.1255, c3-c1 0.216, c1-c2 0.7007. The 0.7
# quantile cut (tau = 0.216) put the Eiffel Tower claim into the bamboo cluster, and the
# two representatives were both bamboo claims, so the tower got no query at all.
CLAIMS = [
    {"id": "c1", "text": "Bamboo can grow 91 cm in a day", "subject_entities": ["Bamboo"],
     "importance": 0.9, "check_worthiness": 0.9},
    {"id": "c2", "text": "the fastest growth of all plants", "subject_entities": ["Bamboo", "plants"],
     "importance": 0.8, "check_worthiness": 0.8},
    {"id": "c3", "text": "The Eiffel Tower is 330 metres tall", "subject_entities": ["Eiffel Tower"],
     "importance": 0.45, "check_worthiness": 0.7},
    {"id": "c4", "text": "Kyiv population exceeds 2.9 million", "subject_entities": ["Kyiv"],
     "importance": 0.4, "check_worthiness": 0.7},
]
GRAPH = GraphResult(mst_edges=[("c4", "c3", 0.1255), ("c3", "c1", 0.216), ("c1", "c2", 0.7007)])
LOOKUP = {c["id"]: c for c in CLAIMS}


def test_trace_scenario_eiffel_claim_becomes_a_representative():
    clusters = build_claim_clusters(claims=CLAIMS, graph_result=GRAPH, quantile=0.7,
                                    representative_min_k=2, representative_max_k=3)
    mixed = next(c for c in clusters if "c3" in c.claim_ids)
    assert set(mixed.claim_ids) == {"c1", "c2", "c3"}  # the grouping itself is unchanged
    assert "c3" in mixed.representative_claim_ids
    assert {"bamboo", "plants", "eiffel tower"} <= set(mixed.covered_entities)


def test_set_cover_is_minimal_then_topped_up_to_floor():
    reps, covered = select_representatives(["c1", "c2", "c3"], LOOKUP, k_floor=1)
    # c2 covers {bamboo, plants} (2 new) before c1 (1); c3 adds {eiffel tower}; c1 adds nothing
    assert reps == ("c2", "c3")
    assert covered == ("bamboo", "eiffel tower", "plants")


def test_floor_preserves_previous_behaviour_when_already_covered():
    reps, _ = select_representatives(["c1", "c2"], LOOKUP, k_floor=2)
    assert set(reps) == {"c1", "c2"}


def test_unverifiable_claims_are_not_representatives():
    lookup = dict(LOOKUP)
    lookup["op"] = {"id": "op", "subject_entities": ["Bamboo beauty"], "verification_target": "none"}
    reps, covered = select_representatives(["op", "c1"], lookup, k_floor=1)
    assert reps == ("c1",) and "bamboo beauty" not in covered


def test_entities_are_normalised():
    lookup = {"a": {"subject_entities": ["Eiffel  Tower"]}, "b": {"subject_entities": ["eiffel tower"]}}
    reps, covered = select_representatives(["a", "b"], lookup, k_floor=1)
    assert reps == ("a",) and covered == ("eiffel tower",)


def test_claims_without_entities_fall_back_to_ranked_floor():
    lookup = {"a": {}, "b": {}}
    assert select_representatives(["a", "b"], lookup, k_floor=1)[0] == ("a",)
