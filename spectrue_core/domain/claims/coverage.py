from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectrue_core.utils.trace import Trace
from .anchors import Anchor, get_anchor_ids, anchors_to_prompt_context


@dataclass
class CoverageGap:
    """Represents uncovered anchors after skeleton extraction."""

    missing_anchor_ids: set[str]
    missing_by_kind: dict[str, int]


# ================= Coverage Validation =================

def validate_coverage(
    anchors: list[Anchor],
    skeleton_items: list[dict[str, Any]],
    skipped_anchors: list[dict[str, Any]],
) -> CoverageGap | None:
    all_anchor_ids = get_anchor_ids(anchors)

    covered_ids: set[str] = set()
    for item in skeleton_items:
        anchor_refs = item.get("anchor_refs", [])
        if isinstance(anchor_refs, list):
            covered_ids.update(anchor_refs)

    skipped_ids: set[str] = set()
    for skip in skipped_anchors:
        anchor_id = skip.get("anchor_id")
        if anchor_id:
            skipped_ids.add(anchor_id)

    missing_ids = all_anchor_ids - covered_ids - skipped_ids

    if not missing_ids:
        return None

    missing_by_kind: dict[str, int] = {}
    for anchor in anchors:
        if anchor.anchor_id in missing_ids:
            kind = anchor.kind.value
            missing_by_kind[kind] = missing_by_kind.get(kind, 0) + 1

    Trace.event(
        "claims.coverage.gaps",
        {
            "missing_anchor_ids": sorted(missing_ids),
            "missing_count": len(missing_ids),
            "missing_by_kind": missing_by_kind,
        },
    )

    return CoverageGap(
        missing_anchor_ids=missing_ids,
        missing_by_kind=missing_by_kind,
    )


def build_gapfill_prompt(
    text: str,
    current_skeleton: dict[str, Any],
    missing_anchors: list[Anchor],
) -> str:
    missing_context = anchors_to_prompt_context(missing_anchors)

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
    original_skeleton: dict[str, Any],
    gapfill_result: dict[str, Any],
) -> dict[str, Any]:
    merged = {
        "events": list(original_skeleton.get("events", [])),
        "measurements": list(original_skeleton.get("measurements", [])),
        "quotes": list(original_skeleton.get("quotes", [])),
        "policies": list(original_skeleton.get("policies", [])),
        "skipped_anchors": list(original_skeleton.get("skipped_anchors", [])),
    }

    for category in ["events", "measurements", "quotes", "policies"]:
        new_key = f"new_{category}"
        new_items = gapfill_result.get(new_key, [])
        if isinstance(new_items, list):
            merged[category].extend(new_items)

    additional_skipped = gapfill_result.get("additional_skipped_anchors", [])
    if isinstance(additional_skipped, list):
        merged["skipped_anchors"].extend(additional_skipped)

    return merged


def check_remaining_gaps(
    anchors: list[Anchor],
    merged_skeleton: dict[str, Any],
) -> list[str]:
    all_items = (
        merged_skeleton.get("events", [])
        + merged_skeleton.get("measurements", [])
        + merged_skeleton.get("quotes", [])
        + merged_skeleton.get("policies", [])
    )

    gap = validate_coverage(
        anchors,
        all_items,
        merged_skeleton.get("skipped_anchors", []),
    )

    if gap is None:
        return []

    Trace.event(
        "claims.coverage.gapfill.incomplete",
        {
            "still_missing": sorted(gap.missing_anchor_ids),
            "count": len(gap.missing_anchor_ids),
        },
    )

    return sorted(gap.missing_anchor_ids)


def emit_coverage_summary(
    anchors: list[Anchor],
    skeleton: dict[str, Any],
    claims_emitted: int,
) -> None:
    all_items = (
        skeleton.get("events", [])
        + skeleton.get("measurements", [])
        + skeleton.get("quotes", [])
        + skeleton.get("policies", [])
    )

    covered_ids: set[str] = set()
    for item in all_items:
        anchor_refs = item.get("anchor_refs", [])
        if isinstance(anchor_refs, list):
            covered_ids.update(anchor_refs)

    skipped_count = len(skeleton.get("skipped_anchors", []))

    Trace.event(
        "claim.coverage.summary",
        {
            "total_anchors": len(anchors),
            "anchors_covered": len(covered_ids),
            "anchors_skipped": skipped_count,
            "claims_emitted": claims_emitted,
        },
    )
