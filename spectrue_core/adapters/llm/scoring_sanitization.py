# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import re

from spectrue_core.utils.security import sanitize_input

MAX_SNIPPET_LEN = 20000
MAX_QUOTE_LEN = 1000


def sanitize_snippet(text: str, *, limit: int = MAX_SNIPPET_LEN) -> str:
    return sanitize_input(text or "")[:limit]


def sanitize_quote(text: str, *, limit: int = MAX_QUOTE_LEN) -> str:
    return sanitize_input(text or "")[:limit]


def format_highlighted_excerpt(*, safe_snippet: str, key_snippet: str | None, stance: str) -> str:
    if key_snippet:
        safe_key = sanitize_quote(key_snippet)
        return f'📌 QUOTE: "{safe_key}"\nℹ️ CONTEXT: {safe_snippet}'
    return safe_snippet


def strip_internal_source_markers(text: str) -> str:
    if not text or not isinstance(text, str):
        return text or ""
    s = text
    s = re.sub(r"\[(?:TRUSTED|RAW)\]\s*", "", s)
    s = re.sub(r"\[REL=\s*\d+(?:\.\d+)?\]\s*", "", s)
    s = re.sub(r"\bTRUSTED\s*/\s*REL=\s*\d+(?:\.\d+)?\b", "", s)
    s = re.sub(r"\bREL=\s*\d+(?:\.\d+)?\b", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def maybe_drop_style_section(rationale: str, *, honesty_score: float | None, lang: str | None) -> str:
    if not rationale or not isinstance(rationale, str):
        return rationale or ""
    try:
        h = float(honesty_score) if honesty_score is not None else None
    except Exception:
        h = None
    if h is None or h < 0.80:
        return rationale

    lc = (lang or "").lower()
    match lc:
        case "uk":
            labels = ["Стиль та контекст:"]
        case "ru":
            labels = ["Стиль и контекст:"]
        case "es":
            labels = ["Estilo y Contexto:"]
        case "de":
            labels = ["Stil und Kontext:"]
        case "fr":
            labels = ["Style et Contexte:"]
        case "ja":
            labels = ["文体と文脈:"]
        case "zh":
            labels = ["风格与语境:"]
        case _:
            labels = ["Style and Context:"]

    s = rationale
    for label in labels:
        s = re.sub(rf"\s*{re.escape(label)}.*$", "", s, flags=re.IGNORECASE | re.DOTALL)
        s = re.sub(rf"(?:^|\n)\s*{re.escape(label)}.*(?:\n|$)", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s
