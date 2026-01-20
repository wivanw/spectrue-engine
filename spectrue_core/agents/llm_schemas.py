# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
LLM structured output schemas used with the Responses API json_schema format.

These schemas are intentionally minimal and aligned to fields consumed by
downstream logic to reduce invalid outputs while preserving fail-soft behavior.
"""

from __future__ import annotations

from typing import Any

from spectrue_core.agents.skills.claims_parsing import ARTICLE_INTENTS
from spectrue_core.schema.claim_metadata import (
    ClaimRole,
    EvidenceChannel,
    MetadataConfidence,
    VerificationTarget,
)
from spectrue_core.schema.claims import ClaimStructureType
from spectrue_core.tools.trusted_sources import AVAILABLE_TOPICS
from spectrue_core.graph.types import EdgeRelation


CLAIM_CATEGORY_VALUES = ["FACTUAL", "SATIRE", "OPINION", "HYPERBOLIC"]
SEARCH_METHOD_VALUES = ["news", "general_search", "academic"]
EVIDENCE_NEED_VALUES = [
    "empirical_study",
    "guideline",
    "official_stats",
    "expert_opinion",
    "anecdotal",
    "news_report",
    "unknown",
]


def _enum_values(enum_cls: type[Any]) -> list[str]:
    return [item.value for item in enum_cls]


CLAIM_ROLE_VALUES = _enum_values(ClaimRole)
VERIFICATION_TARGET_VALUES = _enum_values(VerificationTarget)
EVIDENCE_CHANNEL_VALUES = _enum_values(EvidenceChannel)
METADATA_CONFIDENCE_VALUES = _enum_values(MetadataConfidence)
CLAIM_STRUCTURE_TYPE_VALUES = _enum_values(ClaimStructureType)
EDGE_RELATION_VALUES = _enum_values(EdgeRelation)


SCORING_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_verdicts",
        "verified_score",
        "explainability_score",
        "danger_score",
        "style_score",
        "rationale",
    ],
    "properties": {
        "claim_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_id", "verdict_score", "verdict", "reason", "rgba"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "verified",
                            "refuted",
                            "ambiguous",
                            "unverified",
                            "partially_verified",
                            "plausible",
                            "unlikely",
                        ],
                    },
                    "verdict_score": {"type": "number", "minimum": -1, "maximum": 1},
                    "reason": {"type": "string"},
                    "rgba": {
                        "type": "array",
                        "prefixItems": [
                            {"type": "number", "minimum": 0, "maximum": 1},   # R danger
                            {"type": "number", "minimum": -1, "maximum": 1},  # G verdict_score (can be -1)
                            {"type": "number", "minimum": 0, "maximum": 1},   # B style
                            {"type": "number", "minimum": 0, "maximum": 1},   # A explainability
                        ],
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Per-claim [R=danger, G=veracity, B=style, A=explainability]",
                    },
                    "prior_score": {
                        "type": "number",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "LLM's internal knowledge score for this claim (0-1). 1.0 = known fact, 0.0 = known false, 0.5 = neutral. Use -1 if unknown.",
                    },
                    "prior_reason": {
                        "type": "string",
                        "description": "Short explanation for the prior_score based ONLY on internal knowledge.",
                    },
                },
            },
        },
        "verified_score": {"type": "number", "minimum": 0, "maximum": 1},
        "explainability_score": {"type": "number", "minimum": 0, "maximum": 1},
        "danger_score": {"type": "number", "minimum": 0, "maximum": 1},
        "style_score": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
}


# Schema for scoring a SINGLE claim (used in parallel scoring)
SINGLE_CLAIM_SCORING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_id",
        "verdict_score",
        "verdict",
        "reason",
        "rgba",
    ],
    "properties": {
        "claim_id": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": [
                "verified",
                "refuted",
                "ambiguous",
                "unverified",
                "partially_verified",
                "plausible",
                "unlikely",
            ],
        },
        "verdict_score": {"type": "number", "minimum": -1, "maximum": 1},
        "reason": {"type": "string"},
        "rgba": {
            "type": "array",
            "prefixItems": [
                {"type": "number", "minimum": 0, "maximum": 1},
                {"type": "number", "minimum": -1, "maximum": 1},
                {"type": "number", "minimum": 0, "maximum": 1},
                {"type": "number", "minimum": 0, "maximum": 1},
            ],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[R=danger, G=veracity, B=style, A=explainability]",
        },
        "prior_score": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "LLM's internal knowledge score for this claim (0-1). 1.0 = known fact, 0.0 = known false, 0.5 = neutral. Use -1 if unknown.",
        },
        "prior_reason": {
            "type": "string",
            "description": "Short explanation for the prior_score based ONLY on internal knowledge.",
        },
    },
}


ANALYSIS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verified_score",
        "context_score",
        "danger_score",
        "style_score",
        "confidence_score",
        "rationale",
    ],
    "properties": {
        "verified_score": {"type": "number", "minimum": 0, "maximum": 1},
        "context_score": {"type": "number", "minimum": 0, "maximum": 1},
        "danger_score": {"type": "number", "minimum": 0, "maximum": 1},
        "style_score": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
}


SCORE_EVIDENCE_STRUCTURED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_verdicts",
        "verified_score",
        "explainability_score",
        "danger_score",
        "style_score",
        "rationale",
    ],
    "properties": {
        "claim_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_id", "status", "verdict_score", "reason", "assertion_verdicts"],
                "properties": {
                    "claim_id": {"type": "string"},
                    "status": {"type": "string"},
                    "verdict_score": {"type": "number", "minimum": -1, "maximum": 1},
                    "reason": {"type": "string"},
                    "assertion_verdicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "assertion_key",
                                "dimension",
                                "score",
                                "status",
                                "evidence_count",
                                "rationale",
                            ],
                            "properties": {
                                "assertion_key": {"type": "string"},
                                "dimension": {"type": "string"},
                                "score": {"type": "number", "minimum": 0, "maximum": 1},
                                "status": {"type": "string"},
                                "evidence_count": {"type": "integer", "minimum": 0},
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "verified_score": {"type": "number", "minimum": 0, "maximum": 1},
        "explainability_score": {"type": "number", "minimum": 0, "maximum": 1},
        "danger_score": {"type": "number", "minimum": 0, "maximum": 1},
        "style_score": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
}


QUERY_GENERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["queries", "topics", "claim"],
    "properties": {
        "queries": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 4, "maxLength": 220},
        },
        "topics": {
            "type": "array",
            "items": {"type": "string", "enum": AVAILABLE_TOPICS},
        },
        "claim": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
            },
        },
    },
}


CLAIM_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,  # Allow extra fields (strict=False in API)
    "required": ["claims"],  # Only claims required, article_intent can default to "unknown"
    "properties": {
        "article_intent": {"type": "string", "enum": ARTICLE_INTENTS},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,  # Allow extra fields (strict=False in API)
                "required": [
                    # Core claim fields (must have)
                    "text",
                    "normalized_text",
                    "claim_category",
                    "harm_potential",
                    # Orchestration fields (must have for routing)
                    "verification_target",
                    "claim_role",
                    # The rest are optional - fail-soft with defaults applied downstream:
                    # type, satire_likelihood, importance, check_worthiness,
                    # structure, search_locale_plan, retrieval_policy, metadata_confidence,
                    # query_candidates, search_method, search_queries,
                    # evidence_req, evidence_need, check_oracle
                ],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "normalized_text": {"type": "string", "minLength": 1},
                    "type": {"type": "string"},
                    "claim_category": {"type": "string", "enum": CLAIM_CATEGORY_VALUES},
                    "satire_likelihood": {"type": "number", "minimum": 0, "maximum": 1},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                    "check_worthiness": {"type": "number", "minimum": 0, "maximum": 1},
                    "harm_potential": {"type": "integer", "minimum": 1, "maximum": 5},
                    "verification_target": {
                        "type": "string",
                        "enum": VERIFICATION_TARGET_VALUES,
                    },
                    "claim_role": {"type": "string", "enum": CLAIM_ROLE_VALUES},
                    "structure": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "premises", "conclusion", "dependencies"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": CLAIM_STRUCTURE_TYPE_VALUES,
                            },
                            "premises": {"type": "array", "items": {"type": "string"}},
                            "conclusion": {"type": "string"},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "search_locale_plan": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["primary", "fallback"],
                        "properties": {
                            "primary": {"type": "string"},
                            "fallback": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "retrieval_policy": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["channels_allowed"],
                        "properties": {
                            "channels_allowed": {
                                "type": "array",
                                "items": {"type": "string", "enum": EVIDENCE_CHANNEL_VALUES},
                            },
                        },
                    },
                    "metadata_confidence": {
                        "type": "string",
                        "enum": METADATA_CONFIDENCE_VALUES,
                    },
                    "query_candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "score"],
                            "properties": {
                                "text": {"type": "string"},
                                "role": {"type": "string"},
                                "score": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                    },
                    "search_method": {"type": "string", "enum": SEARCH_METHOD_VALUES},
                    "search_queries": {"type": "array", "items": {"type": "string"}},
                    "evidence_req": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["needs_primary", "needs_2_independent"],
                        "properties": {
                            "needs_primary": {"type": "boolean"},
                            "needs_2_independent": {"type": "boolean"},
                        },
                    },
                    "evidence_need": {"type": "string", "enum": EVIDENCE_NEED_VALUES},
                    "check_oracle": {"type": "boolean"},
                },
            },
        },
    },
}


CORE_CLAIM_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "article_intent": {"type": "string", "enum": ARTICLE_INTENTS},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "normalized_text"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "normalized_text": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


# Verifiable Core Claim Schema
# Enforces verifiability-first contract: every claim must have anchors for search/match/score
PREDICATE_TYPE_VALUES = [
    "event",           # Something happened at a point in time
    "measurement",     # Numeric fact with units
    "policy",          # Official decision/regulation
    "quote",           # Someone said something specific
    "ranking",         # Comparative position (A > B)
    "causal",          # X caused Y
    "existence",       # Entity/document exists with anchors
    "definition",      # Scientific or logical definition
    "property",        # Physical or chemical property
    "other",           # Fallback for edge cases
]

TIME_ANCHOR_TYPE_VALUES = [
    "explicit_date",   # "January 5, 2025"
    "range",           # "between 2020 and 2023"
    "relative",        # "last week", "yesterday"
    "timeless",        # Natural laws, math definitions (no time anchor needed)
    "unknown",         # No time reference found
]

FALSIFIABLE_BY_VALUES = [
    "public_records",        # Government databases, court records
    "scientific_publication", # Peer-reviewed studies, preprints
    "official_statement",    # Press releases, official announcements
    "reputable_news",        # Major news outlets
    "dataset",               # Statistical databases (WHO, Statista, etc.)
    "other",                 # Other verifiable sources
]

EVIDENCE_KIND_VALUES = [
    "primary_source",   # Original document/statement
    "secondary_source", # News coverage, analysis
    "both",             # Either would work
]

LIKELY_SOURCES_VALUES = [
    "authoritative",    # Government, official bodies
    "reputable_news",   # Major news outlets
    "dataset",          # Statistical databases
    "local_media",      # Regional news sources
]

VERIFIABLE_CORE_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {
        "article_intent": {"type": "string", "enum": ARTICLE_INTENTS},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "claim_text",
                    "normalized_text",
                    "subject_entities",
                    "predicate_type",
                    "time_anchor",
                    "falsifiability",
                    "retrieval_seed_terms",
                ],
                "properties": {
                    # Core claim text
                    "claim_text": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": 500,
                        "description": "Single atomic proposition from the article (exact substring)",
                    },
                    "normalized_text": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": 300,
                        "description": "Self-sufficient English version for search queries",
                    },
                    # Entity anchors (required for searchability)
                    "subject_entities": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "maxItems": 12,
                        "description": "Canonical entity names (person, org, place) - required for search",
                    },
                    # Predicate classification
                    "predicate_type": {
                        "type": "string",
                        "enum": PREDICATE_TYPE_VALUES,
                        "description": "Type of factual assertion for routing and scoring",
                    },
                    # Temporal anchor
                    "time_anchor": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "value"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": TIME_ANCHOR_TYPE_VALUES,
                            },
                            "value": {
                                "type": "string",
                                "description": "Time reference as extracted or 'unknown'",
                            },
                        },
                    },
                    # Location anchor (optional but encouraged)
                    "location_anchor": {
                        "type": "string",
                        "description": "Geographic context (city, country, region) or 'unknown'",
                    },
                    # Falsifiability contract
                    "falsifiability": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["is_falsifiable", "falsifiable_by"],
                        "properties": {
                            "is_falsifiable": {
                                "type": "boolean",
                                "description": "True if claim can be proven true/false with evidence",
                            },
                            "falsifiable_by": {
                                "type": "string",
                                "enum": FALSIFIABLE_BY_VALUES,
                                "description": "What type of evidence could verify/refute this claim",
                            },
                        },
                    },
                    # Evidence expectations (optional, relaxed schema)
                    "expected_evidence": {
                        "type": "object",
                        "additionalProperties": True,  # Allow extra fields
                        "properties": {
                            "evidence_kind": {
                                "type": "string",
                                # Removed enum - LLM generates varied values
                            },
                            "likely_sources": {
                                "type": "array",
                                # Removed enum - LLM generates free-form source types
                                "items": {"type": "string"},
                            },
                        },
                    },
                    # Retrieval keywords (NOT full sentences)
                    "retrieval_seed_terms": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 2, "maxLength": 40},
                        "minItems": 3,
                        "maxItems": 15,
                        "description": "Keywords for search, derived from entities + key noun phrases",
                    },
                    # Importance score (kept but not sole gate)
                    "importance": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Claim importance for prioritization (not filtering)",
                    },
                },
            },
        },
    },
}

CLAIM_RETRIEVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    # Relaxed requirements for enrichment failures
    "required": [
        "claim_category",
        "harm_potential",
        "verification_target",
        "claim_role",
    ],
    "properties": {
        "claim_category": {"type": "string", "enum": CLAIM_CATEGORY_VALUES},
        "satire_likelihood": {"type": "number", "minimum": 0, "maximum": 1},
        "importance": {"type": "number", "minimum": 0, "maximum": 1},
        "check_worthiness": {"type": "number", "minimum": 0, "maximum": 1},
        "harm_potential": {"type": "integer", "minimum": 1, "maximum": 5},
        "verification_target": {
            "type": "string",
            "enum": VERIFICATION_TARGET_VALUES,
        },
        "claim_role": {"type": "string", "enum": CLAIM_ROLE_VALUES},
        "structure": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type"],  # Only type is required, rest have defaults
            "properties": {
                "type": {
                    "type": "string",
                    "enum": CLAIM_STRUCTURE_TYPE_VALUES,
                },
                "premises": {"type": "array", "items": {"type": "string"}},
                "conclusion": {"type": "string"},
                "dependencies": {"type": "array", "items": {"type": "string"}},
            },
        },
        "search_locale_plan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primary", "fallback"],
            "properties": {
                "primary": {"type": "string"},
                "fallback": {"type": "array", "items": {"type": "string"}},
            },
        },
        "retrieval_policy": {
            "type": "object",
            "additionalProperties": False,
            "required": ["channels_allowed"],
            "properties": {
                "channels_allowed": {
                    "type": "array",
                    "items": {"type": "string", "enum": EVIDENCE_CHANNEL_VALUES},
                },
            },
        },
        "metadata_confidence": {
            "type": "string",
            "enum": METADATA_CONFIDENCE_VALUES,
        },
        "query_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "score"],
                "properties": {
                    "text": {"type": "string"},
                    "role": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "search_method": {"type": "string", "enum": SEARCH_METHOD_VALUES},
        "search_queries": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "minItems": 1,
            "maxItems": 10,
        },
        "evidence_req": {
            "type": "object",
            "additionalProperties": False,
            "required": ["needs_primary", "needs_2_independent"],
            "properties": {
                "needs_primary": {"type": "boolean"},
                "needs_2_independent": {"type": "boolean"},
            },
        },
        "evidence_need": {"type": "string", "enum": EVIDENCE_NEED_VALUES},
        "check_oracle": {"type": "boolean"},
    },
}

CLAIM_ENRICHMENT_SCHEMA = CLAIM_RETRIEVAL_SCHEMA


EDGE_TYPING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["classifications"],
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "pair_index",
                    "relation",
                    "score",
                    "rationale_short",
                    "evidence_spans",
                ],
                "properties": {
                    "pair_index": {"type": "integer", "minimum": 0},
                    "relation": {"type": "string", "enum": EDGE_RELATION_VALUES},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale_short": {"type": "string"},
                    "evidence_spans": {"type": "string"},
                },
            },
        },
    },
}

# Per-Claim Judging schemas (deep analysis mode)

EVIDENCE_STANCE_VALUES = ["SUPPORT", "REFUTE", "CONTEXT", "IRRELEVANT"]


EVIDENCE_SUMMARIZER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "supporting_evidence",
        "refuting_evidence",
        "contextual_evidence",
        "evidence_gaps",
        "conflicts_present",
    ],
    "properties": {
        "supporting_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "reason"],
                "properties": {
                    "evidence_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "refuting_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "reason"],
                "properties": {
                    "evidence_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "contextual_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "reason"],
                "properties": {
                    "evidence_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
        "evidence_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What evidence is missing to reach a confident verdict",
        },
        "conflicts_present": {
            "type": "boolean",
            "description": "True if evidence contains contradictions",
        },
    },
}


CLAIM_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Per-claim judge output schema (evidence stats are provided in prompt input).",
    "additionalProperties": False,
    "required": [
        "claim_id",
        "rgba",
        "confidence",
        "verdict",
        "explanation",
        "sources_used",
        "missing_evidence",
    ],
    "properties": {
        "claim_id": {"type": "string"},
        "rgba": {
            "type": "object",
            "additionalProperties": False,
            "required": ["R", "G", "B", "A"],
            "properties": {
                "R": {"type": "number", "minimum": -1, "maximum": 1, "description": "Danger (harm if believed). Use -1 if cannot assess."},
                "G": {"type": "number", "minimum": -1, "maximum": 1, "description": "Veracity (factual accuracy). Use -1 for NEI/Unverifiable."},
                "B": {"type": "number", "minimum": -1, "maximum": 1, "description": "Honesty (good faith presentation). Use -1 if cannot assess."},
                "A": {"type": "number", "minimum": -1, "maximum": 1, "description": "Explainability (traceability). Use -1 if no evidence."},
            },
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Overall confidence in the verdict",
        },
        "verdict": {
            "type": "string",
            "enum": ["Supported", "Refuted", "NEI", "Mixed", "Unverifiable"],
            "description": "Final verdict label",
        },
        "explanation": {
            "type": "string",
            "description": "Human-readable explanation of the verdict",
        },
        "prior_score": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "LLM's internal knowledge score for this claim (0-1). 1.0 = known fact, 0.0 = known false, 0.5 = neutral. Use -1 if unknown.",
        },
        "prior_reason": {
            "type": "string",
            "description": "Short explanation for the prior_score based ONLY on internal knowledge.",
        },
        "sources_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "URLs from evidence_items that informed the verdict",
        },
        "missing_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Types of evidence that would strengthen the verdict",
        },
    },
}


CLAIM_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_id",
        "predicate_type",
        "truth_conditions",
        "expected_evidence_types",
        "failure_modes",
        "assertion_strength",
        "risk_facets",
        "honesty_facets",
        "what_would_change_mind",
        "audit_confidence",
    ],
    "properties": {
        "claim_id": {"type": "string"},
        "predicate_type": {
            "type": "string",
            "enum": ["event", "measurement", "quote", "policy", "ranking", "causal", "other"],
        },
        "truth_conditions": {"type": "array", "items": {"type": "string"}},
        "expected_evidence_types": {"type": "array", "items": {"type": "string"}},
        "failure_modes": {"type": "array", "items": {"type": "string"}},
        "assertion_strength": {"type": "string", "enum": ["weak", "medium", "strong"]},
        "risk_facets": {"type": "array", "items": {"type": "string"}},
        "honesty_facets": {"type": "array", "items": {"type": "string"}},
        "what_would_change_mind": {"type": "array", "items": {"type": "string"}},
        "audit_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


EVIDENCE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "claim_id",
        "evidence_id",
        "source_id",
        "stance",
        "directness",
        "specificity",
        "quote_integrity",
        "extraction_confidence",
        "novelty_vs_copy",
        "dependency_hints",
        "audit_confidence",
    ],
    "properties": {
        "claim_id": {"type": "string"},
        "evidence_id": {"type": "string"},
        "source_id": {"type": "string"},
        "stance": {"type": "string", "enum": ["support", "refute", "unclear", "unrelated"]},
        "directness": {"type": "string", "enum": ["direct", "indirect", "tangential"]},
        "specificity": {"type": "string", "enum": ["high", "medium", "low"]},
        "quote_integrity": {"type": "string", "enum": ["ok", "partial", "out_of_context", "not_applicable"]},
        "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "novelty_vs_copy": {"type": "string", "enum": ["original", "syndicated", "unknown"]},
        "dependency_hints": {"type": "array", "items": {"type": "string"}},
        "audit_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


# Coverage Skeleton Schema
# Phase-1 extraction: all events/measurements/quotes/policies with raw_span
# Added anchor_refs for deterministic coverage and skipped_anchors for gap tracking
COVERAGE_SKELETON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["events", "measurements", "quotes", "policies", "skipped_anchors"],
    "properties": {
        # Track anchors that were deliberately skipped
        "skipped_anchors": {
            "type": "array",
            "description": "Anchors not covered by any skeleton item, with reason",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["anchor_id", "reason_code"],
                "properties": {
                    "anchor_id": {"type": "string", "description": "ID of the skipped anchor (t1, n2, q1, etc.)"},
                    "reason_code": {
                        "type": "string",
                        "enum": ["not_a_fact", "duplicate_of", "malformed", "navigation", "boilerplate"],
                        "description": "Why this anchor was not covered",
                    },
                },
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "subject_entities", "verb_phrase", "raw_span"],
                "properties": {
                    "id": {"type": "string"},
                    "subject_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "verb_phrase": {"type": "string", "minLength": 3},
                    "time_anchor": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "value": {"type": "string"},
                        },
                    },
                    "location_anchor": {"type": "string"},
                    "raw_span": {"type": "string", "minLength": 10},
                    # Which deterministic anchors this item covers
                    "anchor_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of anchors (t1, n2, q1) covered by this item",
                    },
                },
            },
        },
        "measurements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "subject_entities", "metric", "quantity_mentions", "raw_span"],
                "properties": {
                    "id": {"type": "string"},
                    "subject_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "metric": {"type": "string", "minLength": 2},
                    "quantity_mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["value"],
                            "properties": {
                                "value": {"type": "string"},
                                "unit": {"type": "string"},
                            },
                        },
                        "minItems": 1,
                    },
                    "time_anchor": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "value": {"type": "string"},
                        },
                    },
                    "raw_span": {"type": "string", "minLength": 10},
                    # Which deterministic anchors this item covers
                    "anchor_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of anchors (t1, n2, q1) covered by this item",
                    },
                },
            },
        },
        "quotes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "speaker_entities", "quote_text", "raw_span"],
                "properties": {
                    "id": {"type": "string"},
                    "speaker_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "quote_text": {"type": "string", "minLength": 5},
                    "raw_span": {"type": "string", "minLength": 10},
                    # Which deterministic anchors this item covers
                    "anchor_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of anchors (t1, n2, q1) covered by this item",
                    },
                },
            },
        },
        "policies": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "subject_entities", "policy_action", "raw_span"],
                "properties": {
                    "id": {"type": "string"},
                    "subject_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "policy_action": {"type": "string", "minLength": 5},
                    "time_anchor": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "value": {"type": "string"},
                        },
                    },
                    "raw_span": {"type": "string", "minLength": 10},
                    # Which deterministic anchors this item covers
                    "anchor_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of anchors (t1, n2, q1) covered by this item",
                    },
                },
            },
        },
    },
}


SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {
    "score_evidence": SCORING_RESPONSE_SCHEMA,
    "score_evidence_structured": SCORE_EVIDENCE_STRUCTURED_SCHEMA,
    "analysis": ANALYSIS_RESPONSE_SCHEMA,
    "query_generation": QUERY_GENERATION_SCHEMA,
    "claim_extraction": CLAIM_EXTRACTION_SCHEMA,
    "core_claim_extraction": CORE_CLAIM_DESCRIPTION_SCHEMA,
    "verifiable_core_claim": VERIFIABLE_CORE_CLAIM_SCHEMA,
    "coverage_skeleton": COVERAGE_SKELETON_SCHEMA,  # M127
    "claim_retrieval_plan": CLAIM_RETRIEVAL_SCHEMA,
    "claim_enrichment": CLAIM_RETRIEVAL_SCHEMA,
    "edge_typing": EDGE_TYPING_SCHEMA,
    "evidence_summarizer": EVIDENCE_SUMMARIZER_SCHEMA,
    "claim_judge": CLAIM_JUDGE_SCHEMA,
    "claim_audit": CLAIM_AUDIT_SCHEMA,
    "evidence_audit": EVIDENCE_AUDIT_SCHEMA,
}


def get_schema(name: str) -> dict[str, Any]:
    return SCHEMA_REGISTRY[name]
