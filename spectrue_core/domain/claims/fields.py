# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Canonical claim field accessors (M146 T010).

Claims carry some fields twice (top level and `metadata`), and readers used
`claim.get(x, 0.5) or 0.5`, which turns a real 0.0 into 0.5. These accessors are
the one way to read those fields: top level first, then `metadata` (dict or
dataclass), normalized, and `None` when the value is absent. Callers decide what
"absent" means; nothing here invents a value.
"""

from __future__ import annotations

from typing import Any, Mapping


def raw(claim: Any, key: str) -> Any:
    """Top-level value, else the `metadata` value (dict or attribute), else None."""
    getter = claim.get if isinstance(claim, Mapping) else (lambda k, _c=claim: getattr(_c, k, None))
    value = getter(key)
    if value is not None:
        return value
    meta = getter("metadata")
    if isinstance(meta, Mapping):
        return meta.get(key)
    return getattr(meta, key, None) if meta is not None else None


def unit(claim: Any, key: str) -> float | None:
    """A [0, 1] number, or None when absent / not numeric. 0.0 stays 0.0."""
    v = raw(claim, key)
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, f))


def number(claim: Any, key: str) -> float | None:
    v = raw(claim, key)
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def enum_value(claim: Any, key: str) -> str | None:
    """Lower-case enum value (accepts Enum members and strings)."""
    v = raw(claim, key)
    if v is None:
        return None
    v = getattr(v, "value", v)
    s = str(v).strip().lower()
    return s or None


def or_default(value: float | None, default: float) -> float:
    """Explicit fallback only when the value is absent (never when it is 0.0)."""
    return default if value is None else value
