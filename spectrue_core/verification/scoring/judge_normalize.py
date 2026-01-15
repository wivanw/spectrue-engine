from __future__ import annotations

from typing import Any


# Canonical stance/verdict enums used by the pipeline.
_STANCE_CANON = {
    "SUPPORT": "SUPPORT",
    "SUPPORTED": "SUPPORT",
    "TRUE": "SUPPORT",
    "CONFIRMED": "SUPPORT",

    "REFUTE": "REFUTE",
    "REFUTED": "REFUTE",
    "FALSE": "REFUTE",
    "DEBUNKED": "REFUTE",

    "MIXED": "MIXED",
    "PARTIAL": "MIXED",
    "PARTLY_TRUE": "MIXED",
    "PARTLY_FALSE": "MIXED",
    "PARTIALLY": "MIXED",

    "NEI": "NEI",
    "UNKNOWN": "NEI",
    "UNVERIFIABLE": "NEI",
    "INSUFFICIENT": "NEI",
    "NOT_ENOUGH_INFO": "NEI",
    "UNCONFIRMED": "NEI",
    "PLAUSIBLE": "NEI",  # treat as uncertain; do not invent confidence
    "UNCLEAR": "NEI",
}


def normalize_verdict_enum(value: Any) -> str:
    """
    Normalize judge verdict/stance labels to canonical enums.
    Never raises. Falls back to NEI.
    """
    if value is None:
        return "NEI"
    s = str(value).strip().upper()
    if not s:
        return "NEI"
    return _STANCE_CANON.get(s, "NEI")


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
    """
    Normalize RGBA into [R,G,B,A] where:
      - R,B,A are clamped to [0,1]
      - G is allowed to be -1.0 or [0,1]
    Returns None if rgba is not usable.
    """
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

    # Preserve signed G semantics: -1.0 means "unverified/NEI".
    if not (g == -1.0 or (0.0 <= g <= 1.0)):
        g = -1.0

    return [r, g, b, a]


def sanitize_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize judge output dict in-place semantics but returns a new dict.
    - Normalizes stance/verdict enums.
    - Normalizes rgba shape without throwing.
    - Never raises.
    """
    out = dict(payload or {})

    # Common field names used across versions
    for key in ("verdict", "stance", "label"):
        if key in out:
            out[key] = normalize_verdict_enum(out.get(key))

    rgba = out.get("rgba")
    nr = normalize_rgba(rgba)
    if nr is not None:
        out["rgba"] = nr

    return out
