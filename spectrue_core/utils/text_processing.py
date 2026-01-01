import re

def clean_search_query(query: str) -> str:
    """
    Sanitize LLM-generated search queries to prevent 'poisonous' searches.
    Removes meta-phrases, fixes broken quotes, and normalizes grammar.
    """
    q = (query or "").strip()

    # 1. Quote normalization (safe structural fix)
    # Replace smart quotes
    q = q.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    # Remove unbalanced double quotes
    if q.count('"') % 2 != 0:
        q = q.replace('"', '')

    # 2. Basic whitespace cleanup
    q = re.sub(r'\s+', ' ', q).strip()

    return q

def normalize_search_query(query: str) -> str:
    """Normalize a search query for consistent caching and search engine behavior."""
    # Use the cleaner first
    q = clean_search_query(query)
    if len(q) > 256:
        # Graceful truncation at last space
        truncated = q[:256]
        if " " in truncated:
            q = truncated.rsplit(" ", 1)[0]
        else:
            q = truncated
    return q.strip()

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
