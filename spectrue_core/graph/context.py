# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import networkx as nx
from typing import List, Optional
from spectrue_core.schema.scoring import ClaimNode, ClaimEdge

class ClaimContextGraph:
    """
    Wrapper around NetworkX DiGraph to manage Claim Context.
    Nodes are claims, Edges are semantic relations (Support, Contradict).
    """
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node: ClaimNode):
        self.graph.add_node(node.claim_id, data=node)

    def add_edge(self, edge: ClaimEdge):
        self.graph.add_edge(
            edge.source_id, 
            edge.target_id, 
            relation=edge.relation, 
            weight=edge.weight
        )

    def get_node(self, claim_id: str) -> Optional[ClaimNode]:
        if self.graph.has_node(claim_id):
            return self.graph.nodes[claim_id]["data"]
        return None

    def get_incoming_edges(self, target_id: str) -> List[ClaimEdge]:
        """
        Returns list of edges pointing to target_id (sources that influence this node).
        """
        edges = []
        if self.graph.has_node(target_id):
            for source, target, data in self.graph.in_edges(target_id, data=True):
                edges.append(ClaimEdge(
                    source_id=source,
                    target_id=target,
                    relation=data["relation"],
                    weight=data["weight"]
                ))
        return edges

    def topological_sort(self) -> List[str]:
        """
        Returns claim IDs in topological order (dependencies first).
        Useful for belief propagation.
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # Cycle detected (Circular reasoning). 
            # Fallback to standard iteration order or break cycles.
            # For robustness, we return nodes in arbitrary order.
            return list(self.graph.nodes)

    @property
    def nodes(self) -> List[ClaimNode]:
        return [self.graph.nodes[n]["data"] for n in self.graph.nodes]
