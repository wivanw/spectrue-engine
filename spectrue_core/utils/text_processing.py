import re

def normalize_search_query(query: str) -> str:
    """Normalize a search query for consistent caching and search engine behavior."""
    q = (query or "").strip()
    q = re.sub(r"\s+", " ", q).strip()
    q = q.strip("“”„«»\"'`")
    q = q.replace("…", " ").strip()
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > 256:
        q = q[:256].strip()
    return q

def clean_article_text(text: str) -> str:
    """
    Remove common boilerplate from article text before processing.
    Helps extract_claims focus on actual content, not navigation/ads.
    """
    if not text:
        return ""
    
    # Common boilerplate patterns to remove
    boilerplate_patterns = [
        r"Read more\s*\.{0,3}",
        r"Share\s+(this|on|via)\s+\w+",
        r"Subscribe\s+(to|for|now)",
        r"Sign up\s+(for|to)",
        r"Follow us\s+on",
        r"Related\s+(articles?|stories|posts)",
        r"Advertisement",
        r"Sponsored\s+content",
        r"Cookie\s+(policy|notice|consent)",
        r"Privacy\s+policy",
        r"Terms\s+(of\s+)?(use|service)",
        r"All\s+rights\s+reserved",
        r"©\s*\d{4}",
        r"\[.*?\]",  # Remove markdown links
        r"#{1,6}\s*$",  # Empty headings
    ]
    
    result = text
    for pattern in boilerplate_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    
    # Collapse multiple newlines/spaces
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r" {2,}", " ", result)
    
    return result.strip()

def extract_claims_heuristic(raw: str, max_claims: int = 2) -> list[str]:
    """Lightweight heuristic claim extraction (sentence splitting)."""
    s = re.sub(r"\s+", " ", (raw or "")).strip()
    if not s:
        return []
    if len(s) <= 260:
        return [s]

    # Lightweight sentence splitting
    parts = re.split(r"(?<=[.!?…])\s+", s)
    candidates = []
    for p in parts:
        p = p.strip()
        if 30 <= len(p) <= 260:
            candidates.append(p)

    if not candidates:
        return [s[:260]]

    def score(sent: str) -> float:
        t = sent
        sc = 0.0
        if re.search(r"\b\d{2,4}\b", t):
            sc += 2.0
        if "%" in t:
            sc += 1.0
        if re.search(r"[$€₴₽]|USD|EUR|UAH|RUB", t):
            sc += 0.5
        sc += min(1.0, len(t) / 180.0)
        return sc

    candidates.sort(key=score, reverse=True)
    out: list[str] = []
    for c in candidates:
        if c not in out:
            out.append(c)
        if len(out) >= max_claims:
            break
    return out

def clamp_score(value) -> float:
    """Clamp value between 0.0 and 1.0."""
    try:
        if value is None:
            return 0.5
        val = float(value)
        return max(0.0, min(1.0, val))
    except (ValueError, TypeError):
        return 0.5

def canonicalize_action(action: str | None) -> str:
    """
    Normalize action verbs to a small canonical set.
    This is used as an explicit anchor for query generation and for downstream scoring heuristics.
    """
    a = (action or "").strip().lower()
    if not a:
        return "claim"

    # Normalize common "strong" verbs first (ban/prohibit/restrict family).
    strong_map = {
        # EN
        "ban": "ban",
        "banned": "ban",
        "prohibit": "prohibit",
        "prohibited": "prohibit",
        "forbid": "forbid",
        "forbidden": "forbid",
        "restrict": "restrict",
        "restricted": "restrict",
        # UK
        "заборона": "ban",
        "заборонили": "ban",
        "заборонено": "ban",
        "заборонити": "ban",
        # RU
        "запрет": "ban",
        "запретили": "ban",
        "запрещено": "ban",
        "запретить": "ban",
    }
    for k, v in strong_map.items():
        if k in a:
            return v

    # A small set of other common actions.
    if any(k in a for k in ("remove", "removed", "take down", "told to remove", "asked to remove")):
        return "remove"
    if any(k in a for k in ("arrest", "arrested", "detain", "detained")):
        return "arrest"
    if any(k in a for k in ("close", "closed", "shut down", "shutdown")):
        return "close"

    # Default: keep a compact cleaned verb phrase (bounded).
    if len(a) > 20:
        return "claim"
    return a
