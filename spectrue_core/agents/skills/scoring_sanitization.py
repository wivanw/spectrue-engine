import re

from spectrue_core.utils.security import sanitize_input

MAX_SNIPPET_LEN = 600
MAX_QUOTE_LEN = 300


def sanitize_snippet(text: str, *, limit: int = MAX_SNIPPET_LEN) -> str:
    return sanitize_input(text or "")[:limit]


def sanitize_quote(text: str, *, limit: int = MAX_QUOTE_LEN) -> str:
    return sanitize_input(text or "")[:limit]


def format_highlighted_excerpt(*, safe_snippet: str, key_snippet: str | None, stance: str) -> str:
    if key_snippet and stance in ["SUPPORT", "REFUTE", "MIXED"]:
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
    if lc == "uk":
        labels = ["Стиль та контекст:"]
    elif lc == "ru":
        labels = ["Стиль и контекст:"]
    elif lc == "es":
        labels = ["Estilo y Contexto:"]
    elif lc == "de":
        labels = ["Stil und Kontext:"]
    elif lc == "fr":
        labels = ["Style et Contexte:"]
    elif lc == "ja":
        labels = ["文体と文脈:"]
    elif lc == "zh":
        labels = ["风格与语境:"]
    else:
        labels = ["Style and Context:"]

    s = rationale
    for label in labels:
        s = re.sub(rf"\s*{re.escape(label)}.*$", "", s, flags=re.IGNORECASE | re.DOTALL)
        s = re.sub(rf"(?:^|\n)\s*{re.escape(label)}.*(?:\n|$)", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

