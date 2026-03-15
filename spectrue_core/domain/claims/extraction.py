"""Deterministic claim extraction helpers and coverage anchoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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


from .anchors import (
    AnchorKind,
    Anchor,
    extract_time_anchors,
    extract_numeric_anchors,
    extract_quote_anchors,
    extract_all_anchors,
    get_anchor_ids,
    anchors_to_prompt_context,
)

from .coverage import (
    CoverageGap,
    validate_coverage,
    build_gapfill_prompt,
    merge_gapfill_result,
    check_remaining_gaps,
    emit_coverage_summary,
)


# ================= Extraction Stats =================


@dataclass
class ExtractionStats:
    """Per-chunk or aggregated stats for claim extraction (emitted vs dropped, reason counts)."""

    claims_extracted_total: int = 0
    claims_dropped_nonverifiable: int = 0
    claims_emitted_targets: int = 0
    drop_reason_counts: dict[str, int] = field(default_factory=dict)

    def record_emit(self) -> None:
        self.claims_emitted_targets += 1

    def record_drop(self, reason_codes: list[str]) -> None:
        self.claims_dropped_nonverifiable += 1
        for r in reason_codes or []:
            self.drop_reason_counts[r] = self.drop_reason_counts.get(r, 0) + 1

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "claims_extracted_total": self.claims_extracted_total,
            "claims_dropped_nonverifiable": self.claims_dropped_nonverifiable,
            "claims_emitted_targets": self.claims_emitted_targets,
            "drop_reason_counts": dict(self.drop_reason_counts),
        }


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
