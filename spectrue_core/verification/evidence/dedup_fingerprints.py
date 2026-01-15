# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import hashlib
import re
from typing import Any


_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_publisher(domain: str) -> str:
    d = (domain or "").strip().lower()
    if d.startswith("www."):
        d = d[4:]
    return d[:96]


def evidence_text_payload(src: dict[str, Any]) -> str:
    """
    Deterministic payload for dedup fingerprints.
    Priority order favors direct spans, then excerpt, then snippet.
    No heuristics; just field precedence.
    """
    for k in ("quote_span", "contradiction_span", "quote", "excerpt", "snippet"):
        v = src.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def normalize_text_for_hash(s: str) -> str:
    s = (s or "").lower()
    s = _NONWORD.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def sha256_hex(s: str) -> str:
    h = hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
    return h


def simhash64(s: str) -> int:
    """
    Classic SimHash 64-bit over token hashes.
    Deterministic and cheap.
    """
    s = normalize_text_for_hash(s)
    if not s:
        return 0
    tokens = s.split(" ")
    v = [0] * 64
    for t in tokens:
        if not t:
            continue
        th = int(hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest(), 16)
        for i in range(64):
            bit = 1 if (th >> i) & 1 else -1
            v[i] += bit
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def simhash_bucket_id(h: int, prefix_bits: int = 16) -> str:
    """
    Near-dup cluster id via prefix bucket. No thresholds.
    """
    if prefix_bits <= 0:
        prefix_bits = 16
    mask = (1 << prefix_bits) - 1
    b = h & mask
    return f"sh{prefix_bits}:{b:04x}"
