from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ClaimPreGraphMeta:
    """Pre-graph metadata computed from embeddings and positional priors."""
    claim_id: str
    position_rank: int
    pos_prior: float
    support_mass: float
    novelty: float
    uncertainty_proxy: float
    importance_prior: float = 0.0
    harm_prior: float = 0.0
    node_prior: float = 0.0


@dataclass
class ClaimPostGraphMeta:
    """Post-graph metadata used for explainability and tracing only."""
    pagerank: float = 0.0
    centrality_rank: int = -1
    selected: bool = False
    selection_gain: float = 0.0
    selection_cost: float = 0.0
    debug: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClaimNode:
    """
    Claim node for graph construction.
    
    Each node must have stable identifiers for traceability:
    claim_id → section_id → anchor
    """
    claim_id: str           # e.g., "c1" (stable identifier)
    text: str               # Claim text (normalized_text preferred)
    claim_type: str         # "core", "numeric", "timeline", "attribution", "sidefact"
    section_id: str         # Section identifier or "main" (for adjacency)
    anchor: str             # Offset or short quote pointer (first 50 chars)
    importance: float       # 0.0-1.0
    topic_key: str          # Topic grouping key
    harm_potential: int = 1 # Harm Potential (1-5)

    # For deduplication
    text_hash: str = ""     # Hash of normalized text for caching

    @classmethod
    def from_claim_dict(cls, claim: dict, index: int = 0) -> "ClaimNode":
        """Create ClaimNode from legacy claim dict."""
        import hashlib

        text = claim.get("normalized_text") or claim.get("text") or ""
        text_hash = hashlib.sha256(text.lower().encode()).hexdigest()[:16]

        return cls(
            claim_id=claim.get("id") or f"c{index + 1}",
            text=text,
            claim_type=claim.get("type", "core"),
            section_id=claim.get("section_id", "main"),
            anchor=text[:50] if text else "",
            importance=float(claim.get("importance", 0.5)),
            topic_key=claim.get("topic_key", "Other"),
            harm_potential=int(claim.get("harm_potential", 1)),
            text_hash=text_hash,
        )
