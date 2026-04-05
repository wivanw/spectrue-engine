from dataclasses import dataclass
from typing import Any
import datetime
import re

@dataclass
class FreshnessSignalRecord:
    claim_id: str
    source_url: str
    parse_status: str  # "parsed" | "missing" | "malformed" | "unsupported"
    parsed_year: int | None
    recency_score: float | None
    contribution_applied: bool
    failure_reason: str | None

def _extract_year(url: str, text: str | None = None) -> int | None:
    """Extract year from URL or text."""
    # Look for explicitly format like /2023/05/
    if url:
        match = re.search(r'/(20\d{2})/', url)
        if match:
            return int(match.group(1))

    # Look for four digits in text/snippet
    if text:
        match = re.search(r'\b(20\d{2})\b', text)
        if match:
            return int(match.group(1))

    return None

def parse_freshness_signal(claim_id: str, item: Any, current_year: int | None = None) -> FreshnessSignalRecord:
    """
    Parse a single evidence item to determine its freshness.
    """
    if current_year is None:
        current_year = datetime.datetime.now().year

    url = getattr(item, 'url', None) or ""
    snippet = getattr(item, 'snippet', None) or ""
    
    if not url and not snippet:
        return FreshnessSignalRecord(
            claim_id=claim_id,
            source_url=url,
            parse_status="missing",
            parsed_year=None,
            recency_score=None,
            contribution_applied=False,
            failure_reason="No URL or snippet to parse",
        )

    parsed_year = _extract_year(url, snippet)
    
    if parsed_year is None:
        return FreshnessSignalRecord(
            claim_id=claim_id,
            source_url=url,
            parse_status="unsupported",
            parsed_year=None,
            recency_score=None,
            contribution_applied=False,
            failure_reason="Could not extract year from content",
        )

    # Valid year found
    age = current_year - parsed_year
    age = max(0, age)  # Just in case parsed year is in the future
    
    # Recency score logic:
    # Linear decay from 0 to 5 years
    # 0 years old: 1.0, 1 year old: 0.8, 2 years old: 0.6, etc.
    recency_score = max(0.0, 1.0 - age * 0.2)
    
    from spectrue_core.utils.trace import Trace
    Trace.event("freshness_signal.parsed", {
        "claim_id": claim_id,
        "source_url": url,
        "parse_status": "parsed",
        "parsed_year": parsed_year,
        "recency_score": recency_score,
    })
    
    return FreshnessSignalRecord(
        claim_id=claim_id,
        source_url=url,
        parse_status="parsed",
        parsed_year=parsed_year,
        recency_score=recency_score,
        contribution_applied=True,
        failure_reason=None,
    )

def calculate_freshness_adjustment(
    claim_id: str, 
    evidence_items: list[Any],
    current_year: int | None = None,
    is_time_sensitive: bool = True
) -> float:
    """
    Calculate an adjustment to confidence based on evidence freshness.
    If all evidence is old, apply a penalty.
    
    If the claim is not time-sensitive (scientific fact, etc.), skip penalty.
    """
    if not is_time_sensitive:
        return 0.0

    if not evidence_items:
        return 0.0

    parsed_records = [parse_freshness_signal(claim_id, item, current_year) for item in evidence_items]
    
    # Only penalize if we actually parsed years and they are all old
    valid_records = [r for r in parsed_records if r.parse_status == "parsed"]
    
    if valid_records:
        average_recency = sum((r.recency_score or 0) for r in valid_records) / len(valid_records)
        # Apply a penalty if average recency is low
        if average_recency < 0.2:
            return -0.2  # 20% confidence reduction
            
    return 0.0
