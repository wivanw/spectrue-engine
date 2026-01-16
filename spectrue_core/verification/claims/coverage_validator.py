# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Coverage Validation and Gap-Fill Repair.

This module validates that all deterministic anchors are either:
- Covered by extracted claims (via anchor_refs)
- Explicitly skipped (via skipped_anchors)

If gaps exist, triggers a targeted gap-fill repair LLM call.
"""

from dataclasses import dataclass

from spectrue_core.utils.trace import Trace
from spectrue_core.verification.claims.coverage_anchors import (
    Anchor,
    get_anchor_ids,
    anchors_to_prompt_context,
)


@dataclass
class CoverageGap:
    """Represents uncovered anchors after skeleton extraction."""
    missing_anchor_ids: set[str]
    missing_by_kind: dict[str, int]  # {"time": 2, "number": 1}


def validate_coverage(
    anchors: list[Anchor],
    skeleton_items: list[dict],
    skipped_anchors: list[dict],
) -> CoverageGap | None:
    """
    Validate that all anchors are covered or skipped.
    
    Args:
        anchors: Deterministic anchors extracted from text
        skeleton_items: Combined list of events, measurements, quotes, policies
        skipped_anchors: List of {anchor_id, reason_code} for intentionally skipped anchors
    
    Returns:
        CoverageGap if any anchors are missing, None if all covered
    """
    all_anchor_ids = get_anchor_ids(anchors)
    
    # Collect covered anchor IDs from skeleton items
    covered_ids: set[str] = set()
    for item in skeleton_items:
        anchor_refs = item.get("anchor_refs", [])
        if isinstance(anchor_refs, list):
            covered_ids.update(anchor_refs)
    
    # Collect skipped anchor IDs
    skipped_ids: set[str] = set()
    for skip in skipped_anchors:
        anchor_id = skip.get("anchor_id")
        if anchor_id:
            skipped_ids.add(anchor_id)
    
    # Compute missing
    missing_ids = all_anchor_ids - covered_ids - skipped_ids
    
    if not missing_ids:
        return None
    
    # Count by kind
    missing_by_kind: dict[str, int] = {}
    for anchor in anchors:
        if anchor.anchor_id in missing_ids:
            kind = anchor.kind.value
            missing_by_kind[kind] = missing_by_kind.get(kind, 0) + 1
    
    # Emit trace event
    Trace.event("claims.coverage.gaps", {
        "missing_anchor_ids": sorted(missing_ids),
        "missing_count": len(missing_ids),
        "missing_by_kind": missing_by_kind,
    })
    
    return CoverageGap(
        missing_anchor_ids=missing_ids,
        missing_by_kind=missing_by_kind,
    )


def build_gapfill_prompt(
    text: str,
    current_skeleton: dict,
    missing_anchors: list[Anchor],
) -> str:
    """
    Build prompt for gap-fill repair LLM call.
    
    Only requests coverage for missing anchors, not re-extraction.
    """
    missing_context = anchors_to_prompt_context(missing_anchors)
    
    # Format current skeleton summary
    skeleton_summary = []
    for category in ["events", "measurements", "quotes", "policies"]:
        items = current_skeleton.get(category, [])
        if items:
            skeleton_summary.append(f"- {category}: {len(items)} items")
    
    return f"""**OUTPUT RULE: Output ONLY valid JSON. No prose. No markdown. Start with {{**

You are performing a GAP-FILL for missing anchors in claim extraction.

## CURRENT EXTRACTION
{chr(10).join(skeleton_summary) if skeleton_summary else "No items extracted yet"}

## MISSING ANCHORS TO COVER
The following anchors were detected in the text but are NOT covered by any extracted item:

{missing_context}

## YOUR TASK
1. For each missing anchor, EITHER:
   - Add a NEW skeleton item that covers it (include it in anchor_refs)
   - Add it to skipped_anchors with a reason_code

2. Do NOT modify existing items
3. Do NOT re-extract already covered anchors

## ORIGINAL TEXT
{text[:3000]}

## OUTPUT FORMAT
{{
  "new_events": [...],
  "new_measurements": [...],
  "new_quotes": [...],
  "new_policies": [...],
  "additional_skipped_anchors": [
    {{"anchor_id": "t1", "reason_code": "not_a_fact"}}
  ]
}}

Valid reason_codes: "not_a_fact", "duplicate_of", "malformed", "navigation", "boilerplate"
"""


def merge_gapfill_result(
    original_skeleton: dict,
    gapfill_result: dict,
) -> dict:
    """
    Merge gap-fill results into original skeleton.
    
    Returns updated skeleton with new items added.
    """
    merged = {
        "events": list(original_skeleton.get("events", [])),
        "measurements": list(original_skeleton.get("measurements", [])),
        "quotes": list(original_skeleton.get("quotes", [])),
        "policies": list(original_skeleton.get("policies", [])),
        "skipped_anchors": list(original_skeleton.get("skipped_anchors", [])),
    }
    
    # Add new items from each category
    for category in ["events", "measurements", "quotes", "policies"]:
        new_key = f"new_{category}"
        new_items = gapfill_result.get(new_key, [])
        if isinstance(new_items, list):
            merged[category].extend(new_items)
    
    # Add additional skipped anchors
    additional_skipped = gapfill_result.get("additional_skipped_anchors", [])
    if isinstance(additional_skipped, list):
        merged["skipped_anchors"].extend(additional_skipped)
    
    return merged


def check_remaining_gaps(
    anchors: list[Anchor],
    merged_skeleton: dict,
) -> list[str]:
    """
    Check if any anchors are still missing after gap-fill.
    
    Returns list of still-missing anchor IDs.
    """
    all_items = (
        merged_skeleton.get("events", []) +
        merged_skeleton.get("measurements", []) +
        merged_skeleton.get("quotes", []) +
        merged_skeleton.get("policies", [])
    )
    
    gap = validate_coverage(
        anchors,
        all_items,
        merged_skeleton.get("skipped_anchors", []),
    )
    
    if gap is None:
        return []
    
    # Log error for remaining gaps (should not happen after gap-fill)
    Trace.event("claims.coverage.gapfill.incomplete", {
        "still_missing": sorted(gap.missing_anchor_ids),
        "count": len(gap.missing_anchor_ids),
    })
    
    return sorted(gap.missing_anchor_ids)


def emit_coverage_summary(
    anchors: list[Anchor],
    skeleton: dict,
    claims_emitted: int,
) -> None:
    """
    Emit final coverage summary trace event.
    """
    all_items = (
        skeleton.get("events", []) +
        skeleton.get("measurements", []) +
        skeleton.get("quotes", []) +
        skeleton.get("policies", [])
    )
    
    # Count covered anchors
    covered_ids: set[str] = set()
    for item in all_items:
        anchor_refs = item.get("anchor_refs", [])
        if isinstance(anchor_refs, list):
            covered_ids.update(anchor_refs)
    
    skipped_count = len(skeleton.get("skipped_anchors", []))
    
    Trace.event("claim.coverage.summary", {
        "total_anchors": len(anchors),
        "anchors_covered": len(covered_ids),
        "anchors_skipped": skipped_count,
        "claims_emitted": claims_emitted,
    })
