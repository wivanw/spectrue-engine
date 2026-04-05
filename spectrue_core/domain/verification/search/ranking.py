from __future__ import annotations

from typing import Iterable


def rerank_search_results(
    results: list[dict],
    *,
    rerank_lambda: float = 0.7,
    top_k: int | None = None,
    skip_extensions: tuple[str, ...] = (".txt", ".xml", ".zip"),
) -> list[dict]:
    """
    Rerank search results using combined score instead of hard filtering.
    
    Formula: score = λ · provider_score + (1-λ) · relevance_score
    """
    from spectrue_core.utils.trace import Trace

    scored: list[tuple[float, dict]] = []

    for r in (results or []):
        url_str = r.get("link", "") or r.get("url", "")
        if isinstance(url_str, str) and url_str.lower().endswith(skip_extensions):
            continue

        provider_score = r.get("score")
        if not isinstance(provider_score, (int, float)):
            provider_score = 0.5
        provider_score = max(0.0, min(1.0, float(provider_score)))

        relevance_score = r.get("relevance_score")
        if not isinstance(relevance_score, (int, float)):
            relevance_score = provider_score
        relevance_score = max(0.0, min(1.0, float(relevance_score)))

        combined = rerank_lambda * provider_score + (1 - rerank_lambda) * relevance_score

        r["_rerank_score"] = combined
        scored.append((combined, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    if top_k is not None and top_k > 0:
        scored = scored[:top_k]

    out = [r for _, r in scored]

    Trace.event(
        "search.rerank",
        {
            "input_count": len(results or []),
            "output_count": len(out),
            "rerank_lambda": rerank_lambda,
            "top_k": top_k,
            "top_scores": [r.get("_rerank_score") for r in out[:3]] if out else [],
        },
    )

    return out


def filter_search_results(
    results: list[dict],
    *,
    min_relevance_score: float = 0.15,
    skip_extensions: tuple[str, ...] = (".txt", ".xml", ".zip"),
) -> list[dict]:
    """
    Legacy filter function - kept for backward compatibility.
    """
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
