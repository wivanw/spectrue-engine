# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# This file is licensed under the GNU Affero General Public License.

from spectrue_core.pipeline.steps.retrieval.cluster_web_search import _assign_similarity_clusters
from spectrue_core.schema.claim_frame import EvidenceItemFrame
from spectrue_core.verification.evidence.evidence_stats import (
    build_evidence_stats,
)
from spectrue_core.verification.scoring.confirmation_counts import compute_confirmation_counts


def test_exact_dupe_counts():
    items = (
        EvidenceItemFrame(evidence_id="e1", claim_id="c1", url="u1", source_id="s1", content_hash="h1"),
        EvidenceItemFrame(evidence_id="e2", claim_id="c1", url="u2", source_id="s2", content_hash="h1"),
        EvidenceItemFrame(evidence_id="e3", claim_id="c1", url="u3", source_id="s3", content_hash="h2"),
    )
    stats = build_evidence_stats(items)
    assert stats.exact_dupes_total == 1


def test_similarity_clusters_stable_ids():
    urls = ["https://a.com/1", "https://b.com/2", "https://c.com/3"]
    sim_matrix = [
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]

    first = _assign_similarity_clusters(urls, sim_matrix, quantile=0.8)
    second = _assign_similarity_clusters(urls, sim_matrix, quantile=0.8)

    assert first == second
    assert first[urls[0]] == first[urls[1]]
    assert first[urls[2]] != first[urls[0]]


def test_confirmation_formula_excludes_duplicates():
    items = (
        EvidenceItemFrame(
            evidence_id="e1",
            claim_id="c1",
            url="u1",
            source_id="s1",
            stance="SUPPORT",
            attribution="precision",
            publisher_id="pub1",
            source_tier="A",
        ),
        EvidenceItemFrame(
            evidence_id="e2",
            claim_id="c1",
            url="u2",
            source_id="s2",
            stance="SUPPORT",
            attribution="precision",
            publisher_id="pub2",
            source_tier="A",
        ),
        EvidenceItemFrame(
            evidence_id="e3",
            claim_id="c1",
            url="u3",
            source_id="s3",
            stance="SUPPORT",
            attribution="corroboration",
            similar_cluster_id="sc1",
            source_tier="A",
        ),
        EvidenceItemFrame(
            evidence_id="e4",
            claim_id="c1",
            url="u4",
            source_id="s4",
            stance="SUPPORT",
            attribution="corroboration",
            similar_cluster_id="sc2",
            source_tier="A",
        ),
    )

    corr = {
        "precision_publishers_support": 2,
        "corroboration_clusters_support": 2,
    }
    counts = compute_confirmation_counts(corr, lam=0.5)
    assert counts["C_precise"] == 2.0
    assert counts["C_corr"] == 2.0
    assert counts["C_total"] == 3.0  # 2 + 0.5 * 2 = 3.0 with lam=0.5
