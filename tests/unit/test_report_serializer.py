# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for claim graph report serializer (compact enum codes)."""

import pytest

from spectrue_core.domain.claims.graph.report_serializer import (
    serialize_graph_for_report,
    RELATION_CODES,
    CLAIM_TYPE_ORDER,
)
from spectrue_core.domain.claims.graph.types import (
    GraphResult,
    RankedClaim,
)
from spectrue_core.domain.claims.graph.nodes import (
    ClaimPreGraphMeta,
    ClaimPostGraphMeta,
)
from spectrue_core.domain.claims.graph.edges import EdgeRelation
from spectrue_core.domain.claims.graph import TypedEdge


@pytest.fixture
def minimal_graph_result():
    pre = {
        "c1": ClaimPreGraphMeta(
            "c1", 1, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
        ),
        "c2": ClaimPreGraphMeta(
            "c2", 2, 0.4, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65
        ),
    }
    post = {
        "c1": ClaimPostGraphMeta(0.25, 0, True, 0.5, 0.1),
        "c2": ClaimPostGraphMeta(0.15, 1, False, 0.2, 0.05),
    }
    ranked = [
        RankedClaim("c1", 0.25, 0.5, 0.0, True),
        RankedClaim("c2", 0.15, 0.3, 0.1, False),
    ]
    edge = TypedEdge(
        "c1", "c2", EdgeRelation.SUPPORTS, 0.9, "rationale", "evidence", False, True
    )
    return GraphResult(
        key_claims=ranked[:1],
        all_ranked=ranked,
        typed_edges=[edge],
        pre_meta=pre,
        post_meta=post,
        disabled=False,
    )


def test_serialize_graph_returns_nodes_and_edges(minimal_graph_result):
    out = serialize_graph_for_report(
        minimal_graph_result,
        {"c1": "text one", "c2": "text two"},
        {"c1": [0.1, 0.2, 0.3, 0.4]},
        {"c1": "core", "c2": "numeric"},
    )
    assert "nodes" in out
    assert "edges" in out
    assert len(out["nodes"]) == 2
    assert len(out["edges"]) == 1


def test_relation_stored_as_numeric_code(minimal_graph_result):
    out = serialize_graph_for_report(
        minimal_graph_result,
        {"c1": "a", "c2": "b"},
    )
    assert out["edges"][0]["relation"] == 0
    assert RELATION_CODES[0] == "supports"


def test_claim_type_stored_as_numeric_code(minimal_graph_result):
    out = serialize_graph_for_report(
        minimal_graph_result,
        {"c1": "a", "c2": "b"},
        claim_id_to_type={"c1": "core", "c2": "sidefact"},
    )
    node_c1 = next(n for n in out["nodes"] if n["claim_id"] == "c1")
    node_c2 = next(n for n in out["nodes"] if n["claim_id"] == "c2")
    assert node_c1.get("claim_type") == 0
    assert node_c2.get("claim_type") == 4
    assert CLAIM_TYPE_ORDER[0] == "core"
    assert CLAIM_TYPE_ORDER[4] == "sidefact"


def test_cross_topic_same_section_as_zero_or_one(minimal_graph_result):
    out = serialize_graph_for_report(
        minimal_graph_result,
        {"c1": "a", "c2": "b"},
    )
    e = out["edges"][0]
    assert e["cross_topic"] in (0, 1)
    assert e["same_section"] in (0, 1)


def test_disabled_graph_returns_empty(minimal_graph_result):
    minimal_graph_result.disabled = True
    out = serialize_graph_for_report(
        minimal_graph_result,
        {"c1": "a"},
    )
    assert out["nodes"] == []
    assert out["edges"] == []


def test_none_graph_returns_empty():
    out = serialize_graph_for_report(None, {})
    assert out["nodes"] == []
    assert out["edges"] == []
