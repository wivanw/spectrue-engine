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

from spectrue_core.agents.skills.claims_parsing import ARTICLE_INTENTS, SEARCH_INTENTS, TOPIC_GROUPS
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
                        ],
                    },
                    "verdict_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                    "rgba": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "Per-claim [R=danger, G=veracity, B=style, A=explainability]",
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
            ],
        },
        "verdict_score": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "rgba": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "minItems": 4,
            "maxItems": 4,
            "description": "[R=danger, G=veracity, B=style, A=explainability]",
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
                    "verdict_score": {"type": "number", "minimum": 0, "maximum": 1},
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
            "minItems": 2,
            "maxItems": 2,
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
    "additionalProperties": False,
    "required": ["article_intent", "claims"],
    "properties": {
        "article_intent": {"type": "string", "enum": ARTICLE_INTENTS},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "text",
                    "normalized_text",
                    "type",
                    "claim_category",
                    "satire_likelihood",
                    "topic_group",
                    "topic_key",
                    "importance",
                    "check_worthiness",
                    "harm_potential",
                    "verification_target",
                    "claim_role",
                    "structure",
                    "search_locale_plan",
                    "retrieval_policy",
                    "metadata_confidence",
                    "search_strategy",
                    "query_candidates",
                    "search_method",
                    "search_queries",
                    "evidence_req",
                    "evidence_need",
                    "check_oracle",
                ],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "normalized_text": {"type": "string", "minLength": 1},
                    "type": {"type": "string"},
                    "claim_category": {"type": "string", "enum": CLAIM_CATEGORY_VALUES},
                    "satire_likelihood": {"type": "number", "minimum": 0, "maximum": 1},
                    "topic_group": {"type": "string", "enum": TOPIC_GROUPS},
                    "topic_key": {"type": "string"},
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
                    "search_strategy": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["intent", "reasoning", "best_language"],
                        "properties": {
                            "intent": {"type": "string", "enum": SEARCH_INTENTS},
                            "reasoning": {"type": "string"},
                            "best_language": {"type": "string"},
                        },
                    },
                    "query_candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "role", "score"],
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


SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {
    "score_evidence": SCORING_RESPONSE_SCHEMA,
    "score_evidence_structured": SCORE_EVIDENCE_STRUCTURED_SCHEMA,
    "analysis": ANALYSIS_RESPONSE_SCHEMA,
    "query_generation": QUERY_GENERATION_SCHEMA,
    "claim_extraction": CLAIM_EXTRACTION_SCHEMA,
    "edge_typing": EDGE_TYPING_SCHEMA,
    "evidence_summarizer": EVIDENCE_SUMMARIZER_SCHEMA,
    "claim_judge": CLAIM_JUDGE_SCHEMA,
}


def get_schema(name: str) -> dict[str, Any]:
    return SCHEMA_REGISTRY[name]

