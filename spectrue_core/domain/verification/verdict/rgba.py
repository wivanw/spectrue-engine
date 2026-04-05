from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class RGBAStatus(str, Enum):
    OK = "OK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    UNVERIFIABLE_BY_NATURE = "UNVERIFIABLE_BY_NATURE"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    EVIDENCE_MISMATCH = "EVIDENCE_MISMATCH"
    """Evidence retrieved is non-empty but not about the claim topic (off-topic)."""


@dataclass
class RGBAMetric:
    status: RGBAStatus
    value: float | None = None
    confidence: float | None = None
    uncertainty: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class RGBAResult:
    R: RGBAMetric
    G: RGBAMetric
    B: RGBAMetric
    A: RGBAMetric
    global_reasons: list[str] = field(default_factory=list)
    summary_trace: dict[str, Any] = field(default_factory=dict)


def clamp_unit(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if v != v:  # NaN
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def normalize_rgba(rgba: Any) -> list[float] | None:
    """Normalize RGBA into [R,G,B,A]."""
    if not isinstance(rgba, list) or len(rgba) != 4:
        return None
    try:
        r = float(rgba[0])
        g = float(rgba[1])
        b = float(rgba[2])
        a = float(rgba[3])
    except Exception:
        return None

    r = clamp_unit(r, 0.0)
    b = clamp_unit(b, 0.0)
    a = clamp_unit(a, 0.0)

    if not (g == -1.0 or (0.0 <= g <= 1.0)):
        g = -1.0

    return [r, g, b, a]
