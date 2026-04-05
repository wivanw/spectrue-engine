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

        # Only trust explicit type if LLM returned something other than default "core"
        explicit_type = claim.get("type")
        if explicit_type and explicit_type != "core":
            claim_type = explicit_type
        else:
            claim_type = _infer_claim_type(claim)

        return cls(
            claim_id=claim.get("id") or f"c{index + 1}",
            text=text,
            claim_type=claim_type,
            section_id=claim.get("section_id", "main"),
            anchor=text[:50] if text else "",
            importance=float(claim.get("importance", 0.5)),
            topic_key=claim.get("topic_key", "Other"),
            harm_potential=int(claim.get("harm_potential", 1)),
            text_hash=text_hash,
        )


# Mapping from claim_role → claim_type for 3D visualization shapes
_ROLE_TO_TYPE: dict[str, str] = {
    "core": "core",
    "thesis": "core",
    "target": "core",
    "counterclaim": "core",
    "attribution": "attribution",
    "aggregated": "attribution",
    "support": "sidefact",
    "subclaim": "sidefact",
    "example": "sidefact",
    "context": "sidefact",
    "background": "sidefact",
    "meta": "sidefact",
    "hedge": "sidefact",
    "definition": "sidefact",
    "forecast": "sidefact",
}


def _infer_claim_type(claim: dict) -> str:
    """Infer claim_type from claim_role, with content heuristics only for 'core' base type."""
    import re

    # 1. Derive from claim_role (authoritative source)
    role = str(
        claim.get("claim_role")
        or claim.get("role")
        or (claim.get("metadata") or {}).get("claim_role")
        or ""
    ).lower().strip()
    base_type = _ROLE_TO_TYPE.get(role, "core")

    # 2. Content heuristics ONLY refine "core" base type into subtypes.
    #    Non-core roles (sidefact, attribution) are NOT overridden by heuristics.
    if base_type != "core":
        return base_type

    text = str(claim.get("normalized_text") or claim.get("text") or "")

    if len(text) < 30:
        return "sidefact"

    # Timeline: has time_anchor or temporal keywords
    if claim.get("time_anchor"):
        return "timeline"
    if re.search(r"\b\d{4}\b", text):
        return "timeline"

    # Numeric: has numbers with units or percentages
    if re.search(r"\b\d+[\.,]?\d*\s*[%$€£₴]", text):
        return "numeric"
    if re.search(r"\b\d+[\.,]\d+\b", text) and re.search(r"(?:млн|тис|billion|million|thousand|percent)", text, re.IGNORECASE):
        return "numeric"

    return base_type
