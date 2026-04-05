from __future__ import annotations

from collections import defaultdict

from .types import ClaimNode, DedupeResult


def deduplicate_claims(nodes: list[ClaimNode]) -> DedupeResult:
    """
    Cluster near-duplicate claims using 90% Jaccard word overlap.
    """
    if not nodes:
        return DedupeResult([], {}, 1.0)

    groups: dict[str, list[ClaimNode]] = defaultdict(list)

    for node in nodes:
        key = " ".join(node.text.lower().split())
        groups[key].append(node)

    canonical_claims: list[ClaimNode] = []
    dedup_map: dict[str, list[str]] = {}

    for _, group in groups.items():
        if len(group) == 1:
            canonical_claims.append(group[0])
            continue

        group.sort(key=lambda n: n.importance, reverse=True)
        canonical = group[0]
        canonical_claims.append(canonical)

        merged_ids = [n.claim_id for n in group[1:]]
        if merged_ids:
            dedup_map[canonical.claim_id] = merged_ids

    canonical_claims = fuzzy_dedup(canonical_claims, threshold=0.9)
    reduction = len(nodes) / len(canonical_claims) if canonical_claims else 1.0

    return DedupeResult(canonical_claims, dedup_map, reduction)


def fuzzy_dedup(
    nodes: list[ClaimNode],
    threshold: float = 0.9,
) -> list[ClaimNode]:
    """Fuzzy deduplication using Jaccard word similarity."""
    if len(nodes) <= 1:
        return nodes

    sorted_nodes = sorted(nodes, key=lambda n: n.importance, reverse=True)
    kept: list[ClaimNode] = []

    for node in sorted_nodes:
        node_words = set(node.text.lower().split())
        if not node_words:
            continue

        is_duplicate = False
        for existing in kept:
            existing_words = set(existing.text.lower().split())
            if not existing_words:
                continue

            intersection = len(node_words & existing_words)
            union = len(node_words | existing_words)
            similarity = intersection / union if union > 0 else 0

            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(node)

    return kept
