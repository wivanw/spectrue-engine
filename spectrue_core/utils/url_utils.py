# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from urllib.parse import urlparse
import re

def get_registrable_domain(url: str) -> str | None:
    """
    Best-effort registrable domain extraction without external deps.
    This is intentionally approximate and conservative.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        host = (urlparse(url).netloc or "").lower().strip()
    except Exception:
        return None
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    # Drop port if present.
    host = host.split(":")[0].strip()
    if not host or "." not in host:
        return None

    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return None

    # Minimal set of common 2-level public suffixes seen in our traffic.
    # Heuristic for 2-level TLDs (e.g. co.uk, com.ua):
    if len(parts) >= 3:
        last = parts[-1]
        second_last = parts[-2]
        if len(last) == 2 and len(second_last) <= 3:
            return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def is_mixed_script(text: str) -> bool:
    """Detect if text contains both Latin and Cyrillic characters (potential evasion)."""
    s = text or ""
    has_latin = re.search(r"[A-Za-z]", s) is not None
    has_cyr = re.search(r"[А-Яа-яІіЇїЄєҐґ]", s) is not None
    return has_latin and has_cyr
