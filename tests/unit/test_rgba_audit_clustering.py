# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from spectrue_core.verification.scoring.rgba_audit.aggregation import cluster_sources
from spectrue_core.verification.scoring.rgba_audit.config import RGBAAuditConfig


def test_redundancy_clustering_is_reproducible():
    sources = [
        {
            "source_id": "s1",
            "title": "City bans fireworks after incidents",
            "snippet": "The city banned fireworks due to safety concerns.",
            "tier": "A",
        },
        {
            "source_id": "s2",
            "title": "City bans fireworks after incidents",
            "snippet": "The city banned fireworks due to safety concerns.",
            "tier": "B",
        },
        {
            "source_id": "s3",
            "title": "Sports update: local team wins",
            "snippet": "The local team won the match last night.",
            "tier": "C",
        },
    ]

    config = RGBAAuditConfig(redundancy_jaccard_threshold=0.8)
    clusters_first = cluster_sources(sources, config)
    clusters_second = cluster_sources(sources, config)

    assert clusters_first == clusters_second
    assert len(clusters_first) == 2

    cluster_sizes = sorted(cluster.size for cluster in clusters_first)
    assert cluster_sizes == [1, 2]

    representative_ids = {cluster.representative_source_id for cluster in clusters_first}
    assert "s1" in representative_ids or "s2" in representative_ids
