# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for claim_graph report serialization."""

from __future__ import annotations

from spectrue_core.domain.claims.graph.report_serializer import (
    DEFAULT_MAX_EVIDENCE_CHARS,
    DEFAULT_MAX_RATIONALE_CHARS,
    fallback_edges_for_nodes,
)


def test_fallback_edges_star_from_key_claim() -> None:
    nodes = [
        {"claim_id": "hub", "is_key_claim": 1},
        {"claim_id": "a", "is_key_claim": 0},
        {"claim_id": "b", "is_key_claim": 0},
    ]
    edges = fallback_edges_for_nodes(nodes)
    assert len(edges) == 2
    assert all(e["src_id"] == "hub" for e in edges)
    assert {e["dst_id"] for e in edges} == {"a", "b"}
    assert all(e.get("synthetic") == 1 for e in edges)
    assert edges[0]["relation"] == 3  # elaborates


def test_fallback_edges_no_key_uses_first_node() -> None:
    nodes = [
        {"claim_id": "x", "is_key_claim": 0},
        {"claim_id": "y", "is_key_claim": 0},
    ]
    edges = fallback_edges_for_nodes(nodes)
    assert len(edges) == 1
    assert edges[0]["src_id"] == "x"
    assert edges[0]["dst_id"] == "y"


def test_fallback_edges_single_node() -> None:
    assert fallback_edges_for_nodes([{"claim_id": "only"}]) == []


def test_default_char_caps_reasonable() -> None:
    assert DEFAULT_MAX_RATIONALE_CHARS >= 500
    assert DEFAULT_MAX_EVIDENCE_CHARS >= 500
