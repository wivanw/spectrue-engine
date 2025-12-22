from __future__ import annotations

import re
from urllib.parse import urlparse

from spectrue_core.tools.url_utils import canonical_url_for_dedupe, is_valid_public_http_url


def clean_tavily_results(results: list[dict] | None) -> list[dict]:
    cleaned: list[dict] = []
    seen: set[str] = set()

    for obj in (results or []):
        title = (obj.get("title") or "").strip()
        url = (obj.get("url") or "").strip()
        content = (obj.get("content") or "").strip()
        raw_content = (obj.get("raw_content") or "").strip()
        score = obj.get("score")

        if not is_valid_public_http_url(url):
            continue

        if not title:
            try:
                title = urlparse(url).netloc or url
            except Exception:
                title = url

        clean_url = canonical_url_for_dedupe(url)
        if clean_url in seen:
            continue
        seen.add(clean_url)

        use_raw = False
        chosen = content
        if raw_content and len(raw_content) > max(200, len(content)):
            use_raw = True
            chosen = raw_content
        chosen = re.sub(r"\s+", " ", chosen).strip()
        if len(chosen) > 2500:
            chosen = chosen[:2500]

        cleaned.append(
            {
                "title": title,
                "url": url,
                "content": chosen,
                "score": score,
                "raw_content_used": use_raw,
            }
        )

    return cleaned


def format_context_from_ranked(ranked: list[dict]) -> str:
    def format_source(obj: dict) -> str:
        return f"Source: {obj['title']}\nURL: {obj['url']}\nContent: {obj['content']}\n---"

    return "\n".join([format_source(obj) for obj in (ranked or [])])


def build_sources_list(
    ranked: list[dict],
    *,
    is_trusted_url,
) -> list[dict]:
    return [
        {
            "title": obj["title"],
            "link": obj["url"],
            "snippet": obj["content"],
            "relevance_score": obj.get("relevance_score"),
            "is_trusted": bool(is_trusted_url(obj.get("url", ""))),
            "origin": "WEB",
        }
        for obj in (ranked or [])
    ]

