# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Schema Mismatch Logger

Centralized logging for cases when LLM responses don't match expected schema.
This helps identify prompt/schema issues and track fallback usage.
"""

from __future__ import annotations

import logging
from typing import Any

from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


def log_field_fallback(
    *,
    skill_name: str,
    item_id: str | None,
    field_name: str,
    expected_type: str,
    received_value: Any,
    fallback_value: Any,
    context: dict[str, Any] | None = None,
) -> None:
    """
    Log when a field uses fallback value because LLM didn't provide expected format.
    
    Args:
        skill_name: Name of the skill (e.g., "claim_extraction", "scoring")
        item_id: ID of the item being processed (e.g., claim_id)
        field_name: Name of the field that was missing/invalid
        expected_type: Expected type description
        received_value: What LLM actually returned
        fallback_value: What we're using instead
        context: Additional context data
    """
    Trace.event(
        f"{skill_name}.field_fallback",
        {
            "item_id": item_id,
            "field": field_name,
            "expected_type": expected_type,
            "received_value": _safe_repr(received_value),
            "fallback_value": _safe_repr(fallback_value),
            **(context or {}),
        },
    )


def log_schema_mismatch(
    *,
    skill_name: str,
    item_id: str | None,
    missing_fields: list[str],
    invalid_fields: dict[str, str] | None = None,
    received_keys: list[str] | None = None,
    raw_response: dict | None = None,
) -> None:
    """
    Log when LLM response is missing required fields or has invalid types.
    
    Args:
        skill_name: Name of the skill
        item_id: ID of the item being processed
        missing_fields: List of required fields that are missing
        invalid_fields: Dict of field_name -> error description
        received_keys: Keys that were actually received
        raw_response: Raw LLM response (truncated)
    """
    logger.warning(
        "[%s] Schema mismatch for %s: missing=%s, invalid=%s",
        skill_name,
        item_id,
        missing_fields,
        list((invalid_fields or {}).keys()),
    )

    Trace.event(
        f"{skill_name}.schema_mismatch",
        {
            "item_id": item_id,
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields or {},
            "received_keys": received_keys,
            "raw_response_head": _truncate_dict(raw_response) if raw_response else None,
        },
    )


def log_claim_field_defaults(
    *,
    claim_id: str,
    defaults_used: dict[str, Any],
    claim_text: str | None = None,
) -> None:
    """
    Log when claim extraction uses default values for missing fields.
    
    Args:
        claim_id: ID of the claim
        defaults_used: Dict of field_name -> default_value used
        claim_text: First 100 chars of claim text for context
    """
    if not defaults_used:
        return

    Trace.event(
        "claim_extraction.defaults_used",
        {
            "claim_id": claim_id,
            "defaults_used": defaults_used,
            "claim_text_head": (claim_text[:100] if claim_text else None),
            "default_count": len(defaults_used),
        },
    )


def validate_claim_response(
    raw_claim: dict,
    claim_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Validate a raw claim from LLM and track which fields used defaults.
    
    Returns:
        Tuple of (defaults_used, invalid_fields)
    """
    defaults_used: dict[str, Any] = {}
    invalid_fields: dict[str, str] = {}

    # Required string fields
    if not raw_claim.get("text"):
        invalid_fields["text"] = "missing or empty"

    # Fields that commonly have defaults
    check_fields = {
        "importance": (float, 0.5),
        "check_worthiness": (float, 0.5),
        "harm_potential": (int, 1),
        "satire_likelihood": (float, 0.0),
        "claim_category": (str, "FACTUAL"),
        "topic_group": (str, "Other"),
        "verification_target": (str, "reality"),
        "claim_role": (str, "core"),
        "metadata_confidence": (str, "medium"),
        "search_method": (str, "general_search"),
    }

    for field, (expected_type, default_val) in check_fields.items():
        val = raw_claim.get(field)
        if val is None:
            defaults_used[field] = default_val
        elif expected_type == float and not isinstance(val, (int, float)):
            defaults_used[field] = default_val
            invalid_fields[field] = f"expected float, got {type(val).__name__}"
        elif expected_type == int and not isinstance(val, int):
            defaults_used[field] = default_val
            invalid_fields[field] = f"expected int, got {type(val).__name__}"
        elif expected_type == str and not isinstance(val, str):
            defaults_used[field] = default_val
            invalid_fields[field] = f"expected str, got {type(val).__name__}"

    # Check nested objects
    if not isinstance(raw_claim.get("search_strategy"), dict):
        defaults_used["search_strategy"] = {}

    if not isinstance(raw_claim.get("evidence_req"), dict):
        defaults_used["evidence_req"] = {"needs_primary": False, "needs_2_independent": False}

    if not isinstance(raw_claim.get("search_locale_plan"), dict):
        defaults_used["search_locale_plan"] = {"primary": "en", "fallback": []}

    # Check arrays
    if not isinstance(raw_claim.get("search_queries"), list):
        defaults_used["search_queries"] = []

    if not isinstance(raw_claim.get("query_candidates"), list):
        defaults_used["query_candidates"] = []

    return defaults_used, invalid_fields


def validate_scoring_response(
    result: dict,
    claim_id: str,
) -> tuple[list[str], dict[str, str]]:
    """
    Validate scoring LLM response.
    
    Returns:
        Tuple of (missing_fields, invalid_fields)
    """
    missing_fields: list[str] = []
    invalid_fields: dict[str, str] = {}

    # Required fields
    if result.get("verdict_score") is None:
        missing_fields.append("verdict_score")
    elif not isinstance(result.get("verdict_score"), (int, float)):
        invalid_fields["verdict_score"] = f"expected float, got {type(result.get('verdict_score')).__name__}"

    if not result.get("verdict"):
        missing_fields.append("verdict")
    elif result.get("verdict") not in {"verified", "refuted", "ambiguous", "unverified"}:
        invalid_fields["verdict"] = f"unexpected value: {result.get('verdict')}"

    # RGBA validation
    rgba = result.get("rgba")
    if rgba is None:
        missing_fields.append("rgba")
    elif not isinstance(rgba, list):
        invalid_fields["rgba"] = f"expected list, got {type(rgba).__name__}"
    elif len(rgba) < 4:
        invalid_fields["rgba"] = f"expected 4 elements, got {len(rgba)}"
    elif not all(isinstance(x, (int, float)) for x in rgba[:4]):
        invalid_fields["rgba"] = "elements must be numbers"

    return missing_fields, invalid_fields


def _safe_repr(value: Any, max_len: int = 200) -> str:
    """Safely convert value to string representation, truncated."""
    try:
        s = repr(value)
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s
    except Exception:
        return f"<{type(value).__name__}>"


def _truncate_dict(d: dict | None, max_keys: int = 10) -> dict | None:
    """Truncate dict for logging."""
    if d is None:
        return None
    result = {}
    for i, (k, v) in enumerate(d.items()):
        if i >= max_keys:
            result["..."] = f"{len(d) - max_keys} more keys"
            break
        if isinstance(v, str) and len(v) > 200:
            result[k] = v[:197] + "..."
        elif isinstance(v, list) and len(v) > 5:
            result[k] = v[:5] + [f"... {len(v) - 5} more"]
        elif isinstance(v, dict):
            result[k] = _truncate_dict(v, max_keys=5)
        else:
            result[k] = v
    return result

