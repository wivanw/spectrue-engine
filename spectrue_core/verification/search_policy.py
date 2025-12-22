from __future__ import annotations

from typing import Iterable


def filter_search_results(
    results: list[dict],
    *,
    min_relevance_score: float = 0.15,
    skip_extensions: tuple[str, ...] = (".txt", ".xml", ".zip"),
) -> list[dict]:
    out: list[dict] = []
    for r in (results or []):
        score = r.get("relevance_score")
        if isinstance(score, (int, float)) and float(score) < float(min_relevance_score):
            continue

        url_str = r.get("link", "") or r.get("url", "")
        if isinstance(url_str, str) and url_str.lower().endswith(skip_extensions):
            continue

        out.append(r)
    return out


def should_fallback_news_to_general(topic: str, filtered: list[dict]) -> tuple[bool, str, float]:
    if topic != "news":
        return False, "", 0.0
    valid_count = len(filtered or [])
    max_score = max([float(r.get("score", 0) or 0.0) for r in (filtered or [])]) if filtered else 0.0
    if valid_count < 2:
        return True, f"few_results ({valid_count})", max_score
    if max_score < 0.2:
        return True, f"low_relevance ({max_score:.2f})", max_score
    return False, "", max_score


def build_context_from_sources(sources: Iterable[dict]) -> str:
    def format_source(obj: dict) -> str:
        return f"Source: {obj.get('title')}\nURL: {obj.get('link')}\nContent: {obj.get('snippet')}\n---"

    return "\n".join([format_source(obj) for obj in (sources or [])])


def prefer_fallback_results(
    *,
    original_filtered: list[dict],
    original_max_score: float,
    fallback_filtered: list[dict],
) -> bool:
    fb_count = len(fallback_filtered or [])
    fb_max_score = max([float(r.get("score", 0) or 0.0) for r in (fallback_filtered or [])]) if fallback_filtered else 0.0

    return fb_count > 0 and (len(original_filtered or []) == 0 or fb_max_score > original_max_score)

