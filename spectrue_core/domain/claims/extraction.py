"""Deterministic claim extraction helpers and coverage anchoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from spectrue_core.utils.trace import Trace


# Default system instructions for claim extraction (MANDATORY)
# These MUST always be present in LLM calls to prevent extraction failures.
DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS = """
You are extracting factual claims for verification.
Do NOT summarize. Do NOT judge truth.
Your task is to enumerate verifiable factual units.

Rules:
- Prefer over-extraction to under-extraction.
- Extract events, measurements, quantities, dates, and quoted statements.
- Each claim must be atomic and independently verifiable.
- Output must strictly follow the provided JSON schema.
- For 'article_intent', use ONLY: "news", "evergreen", "official", "opinion", "prediction", "unknown", "other".
"""


# Time anchors are NOT universally required.
# Many predicates are timeless (definitions/properties/measurements) and forcing
# time_anchor makes claims "mathematically unreachable" in non-experiment mode.
TIME_ANCHOR_EXEMPT_PREDICATES = {
    "quote",
    "policy",
    "ranking",
    "existence",
    "measurement",
    "definition",
    "property",
}


class AnchorKind(Enum):
    """Anchor type classification."""

    TIME = "time"
    NUMBER = "number"
    QUOTE = "quote"


@dataclass
class Anchor:
    """
    A deterministic anchor extracted from text.

    Anchors are structural markers (dates, numbers, quotes) that represent
    verifiable facts which must be covered by extracted claims.
    """

    anchor_id: str
    kind: AnchorKind
    span_text: str
    char_start: int
    char_end: int
    context_window: str


@dataclass
class CoverageGap:
    """Represents uncovered anchors after skeleton extraction."""

    missing_anchor_ids: set[str]
    missing_by_kind: dict[str, int]


class ExtractionStats:
    """Track extraction statistics for observability."""

    def __init__(self) -> None:
        self.claims_extracted_total = 0
        self.claims_dropped_nonverifiable = 0
        self.claims_emitted_targets = 0
        self.drop_reason_counts: dict[str, int] = {}

    def record_drop(self, reason_codes: list[str]) -> None:
        self.claims_dropped_nonverifiable += 1
        for reason in reason_codes:
            self.drop_reason_counts[reason] = self.drop_reason_counts.get(reason, 0) + 1

    def record_emit(self) -> None:
        self.claims_emitted_targets += 1

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "claims_extracted_total": self.claims_extracted_total,
            "claims_dropped_nonverifiable": self.claims_dropped_nonverifiable,
            "claims_emitted_targets": self.claims_emitted_targets,
            "drop_reason_counts": self.drop_reason_counts,
        }


# ================= Time Anchors =================

TIME_PATTERNS = [
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", "iso_date"),
    (r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", "eu_date"),
    (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "us_date"),
    (r"\b(\d{4})-(\d{2})\b", "year_month"),
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", "month_year"),
    (r"\b[Qq]([1-4])\s*(\d{4})\b", "quarter"),
    (r"\b(19\d{2}|20\d{2})\b", "year"),
]


def _extract_context_window(text: str, start: int, end: int, window_tokens: int = 5) -> str:
    """Extract ±N tokens around a span."""
    before = text[:start].split()
    after = text[end:].split()

    prefix = " ".join(before[-window_tokens:]) if before else ""
    suffix = " ".join(after[:window_tokens]) if after else ""
    span = text[start:end]

    parts = [p for p in [prefix, span, suffix] if p]
    return " ".join(parts)


def extract_time_anchors(text: str) -> list[Anchor]:
    """Extract time anchors from text using date patterns."""
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0

    for pattern, _pattern_type in TIME_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()

            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue

            counter += 1
            anchors.append(
                Anchor(
                    anchor_id=f"t{counter}",
                    kind=AnchorKind.TIME,
                    span_text=match.group(0),
                    char_start=start,
                    char_end=end,
                    context_window=_extract_context_window(text, start, end),
                )
            )
            seen_spans.add((start, end))

    return anchors


# ================= Numeric Anchors =================

NUMERIC_PATTERNS = [
    (r"[$€£¥]\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|k|m|b))?\b", "currency"),
    (r"\b\d+(?:\.\d+)?\s*%", "percentage"),
    (r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", "large_number"),
    (r"\b\d+(?:\.\d+)?\s*(?:km|m|cm|mm|kg|g|mg|lb|oz|°[CF]|mph|kph)\b", "number_with_unit"),
    (r"\b\d+\.\d+\b", "decimal"),
    (r"\b\d{2,}\b", "integer"),
]


def extract_numeric_anchors(text: str) -> list[Anchor]:
    """Extract numeric anchors from text."""
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0

    for pattern, _pattern_type in NUMERIC_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()

            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue

            counter += 1
            anchors.append(
                Anchor(
                    anchor_id=f"n{counter}",
                    kind=AnchorKind.NUMBER,
                    span_text=match.group(0),
                    char_start=start,
                    char_end=end,
                    context_window=_extract_context_window(text, start, end),
                )
            )
            seen_spans.add((start, end))

    return anchors


# ================= Quote Anchors =================

QUOTE_PATTERNS = [
    (r'"([^"]+)"', "double_quote"),
    (r"'([^']+)'", "single_quote"),
    (r"[\u201c]([^\u201d]+)[\u201d]", "curly_double"),
    (r"[\u2018]([^\u2019]+)[\u2019]", "curly_single"),
    (r"«([^»]+)»", "guillemet"),
    (r'„([^"]+)"', "german_quote"),
]


def extract_quote_anchors(text: str) -> list[Anchor]:
    """Extract quoted spans from text."""
    anchors: list[Anchor] = []
    seen_spans: set[tuple[int, int]] = set()
    counter = 0

    for pattern, _pattern_type in QUOTE_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()

            if end - start < 5:
                continue

            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue

            counter += 1
            anchors.append(
                Anchor(
                    anchor_id=f"q{counter}",
                    kind=AnchorKind.QUOTE,
                    span_text=match.group(0),
                    char_start=start,
                    char_end=end,
                    context_window=_extract_context_window(text, start, end),
                )
            )
            seen_spans.add((start, end))

    return anchors


# ================= Combined Extraction =================


def extract_all_anchors(text: str) -> list[Anchor]:
    """Extract all deterministic anchors from text."""
    if not text:
        return []

    time_anchors = extract_time_anchors(text)
    numeric_anchors = extract_numeric_anchors(text)
    quote_anchors = extract_quote_anchors(text)

    all_anchors = time_anchors + numeric_anchors + quote_anchors
    all_anchors.sort(key=lambda a: a.char_start)

    Trace.event(
        "claims.coverage.anchors",
        {
            "counts": {
                "time": len(time_anchors),
                "number": len(numeric_anchors),
                "quote": len(quote_anchors),
                "total": len(all_anchors),
            },
            "anchors": [
                {
                    "anchor_id": a.anchor_id,
                    "kind": a.kind.value,
                    "preview": a.span_text[:30],
                }
                for a in all_anchors[:20]
            ],
        },
    )

    return all_anchors


def get_anchor_ids(anchors: list[Anchor]) -> set[str]:
    return {a.anchor_id for a in anchors}


def anchors_to_prompt_context(anchors: list[Anchor]) -> str:
    if not anchors:
        return ""

    lines = []
    for a in anchors:
        lines.append(
            f"- [{a.anchor_id}] {a.kind.value}: \"{a.span_text}\" (context: {a.context_window[:60]}...)"
        )

    return "\n".join(lines)


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


# ================= Validation Helpers =================


def validate_core_claim(claim: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Deterministic validation for verifiable claims.
    """
    reason_codes: list[str] = []

    claim_text = claim.get("claim_text") or claim.get("text") or ""

    if not claim_text:
        reason_codes.append("empty_claim_text")
    elif len(claim_text) > 500:
        reason_codes.append("claim_text_too_long")

    entities = claim.get("subject_entities")
    if not entities or not isinstance(entities, list) or len(entities) < 1:
        reason_codes.append("missing_subject_entities")
    else:
        valid_entities = [e for e in entities if isinstance(e, str) and e.strip()]
        if len(valid_entities) < 1:
            reason_codes.append("invalid_subject_entities")

    seed_terms = claim.get("retrieval_seed_terms")
    if not seed_terms or not isinstance(seed_terms, list):
        reason_codes.append("missing_retrieval_seed_terms")
    else:
        valid_terms = [t for t in seed_terms if isinstance(t, str) and len(t) >= 2]
        if len(valid_terms) < 3:
            reason_codes.append("insufficient_retrieval_seed_terms")
        elif len(valid_terms) > 10:
            reason_codes.append("too_many_retrieval_seed_terms")

    falsifiability = claim.get("falsifiability")
    if not falsifiability or not isinstance(falsifiability, dict):
        reason_codes.append("missing_falsifiability")
    else:
        is_falsifiable = falsifiability.get("is_falsifiable")
        if is_falsifiable is not True:
            reason_codes.append("not_falsifiable")

    predicate_type = claim.get("predicate_type", "other")
    time_anchor = claim.get("time_anchor")

    if predicate_type not in TIME_ANCHOR_EXEMPT_PREDICATES:
        if not time_anchor or not isinstance(time_anchor, dict):
            reason_codes.append("missing_time_anchor")
        else:
            anchor_type = time_anchor.get("type", "unknown")
            if anchor_type == "unknown":
                reason_codes.append("unknown_time_anchor")

    ok = len(reason_codes) == 0
    return ok, reason_codes


def extract_keywords_deterministic(text: str, max_tokens: int = 6) -> str:
    """
    Extract keyword query from claim text (deterministic, language-agnostic).
    """
    if not text:
        return ""

    lowered = text.lower()
    cleaned = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    tokens = cleaned.split()

    unique_tokens: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if len(t) >= 3 and t not in seen:
            unique_tokens.append(t)
            seen.add(t)
        if len(unique_tokens) >= max_tokens:
            break

    if unique_tokens:
        return " ".join(unique_tokens)

    return text[:80].rstrip(".,!?;")
