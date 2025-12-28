from __future__ import annotations

import hashlib
import statistics
import string
from dataclasses import dataclass
from typing import List, Optional

from spectrue_core.runtime_config import ContentBudgetConfig


_PUNCTUATION = set(string.punctuation)


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _simhash(text: str, hashbits: int = 64) -> int:
    """
    Lightweight SimHash implementation on lowercased word tokens.
    Deterministic and dependency-free.
    """
    if not text:
        return 0
    tokens = [t for t in text.lower().split() if t]
    if not tokens:
        return 0
    v = [0] * hashbits
    for token in tokens:
        h = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)
        for i in range(hashbits):
            bit = 1 if (h >> i) & 1 else -1
            v[i] += bit
    fingerprint = 0
    for i, weight in enumerate(v):
        if weight >= 0:
            fingerprint |= 1 << i
    return fingerprint


@dataclass(frozen=True)
class BlockFeatures:
    index: int
    text: str
    len_chars: int
    unique_char_ratio: float
    digit_ratio: float
    punctuation_ratio: float
    whitespace_ratio: float
    simhash: int
    weight: float


@dataclass(frozen=True)
class TrimResult:
    trimmed_text: str
    selection_meta: List[dict]
    raw_len: int
    trimmed_len: int
    raw_sha256: str
    trimmed_sha256: str
    blocks_stats: dict
    budget_limit: int
    budget_used: int

    def trace_blocks_payload(self) -> dict:
        payload = dict(self.blocks_stats)
        payload.update(
            {
                "raw_len": self.raw_len,
                "trimmed_len": self.trimmed_len,
                "budget_limit": self.budget_limit,
                "budget_used": self.budget_used,
            }
        )
        return payload

    def trace_selection_payload(self, trace_top_blocks: int) -> dict:
        return {
            "selected": self.selection_meta[: max(0, trace_top_blocks)],
            "raw_len": self.raw_len,
            "trimmed_len": self.trimmed_len,
            "raw_sha256": self.raw_sha256,
            "trimmed_sha256": self.trimmed_sha256,
            "budget_limit": self.budget_limit,
            "budget_used": self.budget_used,
        }


class ContentBudgeter:
    """
    Deterministic text budgeter for already-clean plain text.
    Splits into structural blocks, scores with lightweight statistics,
    and greedily selects diverse blocks under a character budget.
    """

    def __init__(self, config: Optional[ContentBudgetConfig] = None):
        self.config = config or ContentBudgetConfig()
        self._hashbits = 64

    def trim(self, text: str, *, budget_limit: Optional[int] = None) -> TrimResult:
        raw_text = (text or "").strip()
        raw_len = len(raw_text)

        if raw_len == 0:
            return TrimResult(
                trimmed_text="",
                selection_meta=[],
                raw_len=0,
                trimmed_len=0,
                raw_sha256=_sha256(""),
                trimmed_sha256=_sha256(""),
                blocks_stats={"block_count": 0, "selected_count": 0, "len_min": 0, "len_mean": 0, "len_max": 0},
                budget_limit=budget_limit or self.config.max_clean_text_chars_default,
                budget_used=0,
            )

        guardrail = int(self.config.absolute_guardrail_chars)
        if raw_len > guardrail:
            raise ValueError(f"Input too large for budgeting (>{guardrail} chars)")

        budget = int(budget_limit or self._resolve_budget(raw_len))
        blocks = self._split_into_blocks(raw_text)
        block_lengths = [b.len_chars for b in blocks] or [raw_len]
        blocks_stats = {
            "block_count": len(blocks),
            "selected_count": 0,
            "len_min": min(block_lengths),
            "len_mean": statistics.fmean(block_lengths) if block_lengths else 0,
            "len_max": max(block_lengths),
        }

        if raw_len <= budget:
            return TrimResult(
                trimmed_text=raw_text,
                selection_meta=[],
                raw_len=raw_len,
                trimmed_len=raw_len,
                raw_sha256=_sha256(raw_text),
                trimmed_sha256=_sha256(raw_text),
                blocks_stats=blocks_stats,
                budget_limit=budget,
                budget_used=raw_len,
            )

        selected, budget_used, selection_meta = self._select_blocks(blocks, budget)
        blocks_stats["selected_count"] = len(selected)

        if not selected:
            candidates = [b for b in blocks if b.len_chars <= budget] or blocks
            fallback = max(candidates, key=lambda b: (b.weight, -b.index))
            selected = [fallback]
            budget_used = fallback.len_chars
            selection_meta = [
                {
                    "index": fallback.index,
                    "len": fallback.len_chars,
                    "unique_ratio": fallback.unique_char_ratio,
                    "digit_ratio": fallback.digit_ratio,
                    "punct_ratio": fallback.punctuation_ratio,
                    "whitespace_ratio": fallback.whitespace_ratio,
                    "simhash": format(fallback.simhash, "016x"),
                    "weight": round(fallback.weight, 4),
                    "coverage_gain": 1.0,
                    "score": round(fallback.weight, 4),
                    "reason": "fallback_strongest",
                }
            ]
            blocks_stats["selected_count"] = 1

        selected_sorted = sorted(selected, key=lambda b: b.index)
        trimmed_text = "\n\n".join([b.text for b in selected_sorted]).strip()
        trimmed_len = len(trimmed_text)

        return TrimResult(
            trimmed_text=trimmed_text,
            selection_meta=selection_meta,
            raw_len=raw_len,
            trimmed_len=trimmed_len,
            raw_sha256=_sha256(raw_text),
            trimmed_sha256=_sha256(trimmed_text),
            blocks_stats=blocks_stats,
            budget_limit=budget,
            budget_used=budget_used,
        )

    def _resolve_budget(self, raw_len: int) -> int:
        if raw_len > int(self.config.max_clean_text_chars_huge_input):
            return int(self.config.max_clean_text_chars_huge_input)
        return int(self.config.max_clean_text_chars_default)

    def _split_into_blocks(self, text: str) -> List[BlockFeatures]:
        blocks: List[BlockFeatures] = []
        current_lines: List[str] = []
        index = 0
        lines = text.splitlines()
        for line in lines:
            stripped = line.strip()
            is_heading = stripped.startswith("#")
            if not stripped:
                if current_lines:
                    blocks.append(self._build_block("\n".join(current_lines), index))
                    index += 1
                    current_lines = []
                continue
            if is_heading and current_lines:
                blocks.append(self._build_block("\n".join(current_lines), index))
                index += 1
                current_lines = [stripped]
                continue
            current_lines.append(stripped)

        if current_lines:
            blocks.append(self._build_block("\n".join(current_lines), index))

        return blocks

    def _build_block(self, text: str, index: int) -> BlockFeatures:
        length = len(text)
        if length == 0:
            # Avoid division by zero in ratios
            return BlockFeatures(
                index=index,
                text="",
                len_chars=0,
                unique_char_ratio=0.0,
                digit_ratio=0.0,
                punctuation_ratio=0.0,
                whitespace_ratio=0.0,
                simhash=0,
                weight=0.0,
            )

        unique_ratio = len(set(text)) / length
        digit_ratio = sum(ch.isdigit() for ch in text) / length
        punctuation_ratio = sum(ch in _PUNCTUATION for ch in text) / length
        whitespace_ratio = sum(ch.isspace() for ch in text) / length

        # Structural weight favors longer, denser, more varied blocks deterministically
        density = max(0.0, 1 - whitespace_ratio)
        variety = unique_ratio
        weight = length * ((density + variety) / 2)

        return BlockFeatures(
            index=index,
            text=text,
            len_chars=length,
            unique_char_ratio=unique_ratio,
            digit_ratio=digit_ratio,
            punctuation_ratio=punctuation_ratio,
            whitespace_ratio=whitespace_ratio,
            simhash=_simhash(text, hashbits=self._hashbits),
            weight=weight,
        )

    def _select_blocks(
        self, blocks: List[BlockFeatures], budget: int
    ) -> tuple[List[BlockFeatures], int, List[dict]]:
        """
        Greedy facility-location objective:
        maximize coverage = sum_i weight_i * max_{j in S} dist_norm(i, j)
        under character budget, keeping deterministic order.
        """
        min_chars = int(self.config.block_min_chars)
        usable = [b for b in blocks if b.len_chars >= min_chars]
        if not usable:
            return [], 0, []

        n = len(usable)
        similarities: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i, bi in enumerate(usable):
            for j in range(i, n):
                if i == j:
                    similarities[i][j] = 1.0
                    continue
                dist = _hamming_distance(bi.simhash, usable[j].simhash) / self._hashbits
                sim = max(0.0, 1.0 - dist)
                similarities[i][j] = sim
                similarities[j][i] = sim

        weights = [b.weight for b in usable]
        current_cov = [0.0] * n
        selected_indices: List[int] = []
        budget_used = 0
        selection_meta: List[dict] = []
        selected_text_hashes: set[str] = set()

        remaining_indices = list(range(n))

        while remaining_indices:
            best_idx = None
            best_gain = -1.0
            best_budget_cost = 0
            best_cov_gain = 0.0

            for idx in remaining_indices:
                candidate = usable[idx]
                if budget_used + candidate.len_chars > budget:
                    continue

                if _sha256(candidate.text) in selected_text_hashes:
                    continue

                # Marginal coverage gain
                gain = candidate.weight
                coverage_delta = 0.0
                for i in range(n):
                    new_cov = max(current_cov[i], similarities[i][idx])
                    coverage_delta += (new_cov - current_cov[i]) * weights[i]
                gain += coverage_delta

                if best_idx is None:
                    best_idx = idx
                    best_gain = gain
                    best_budget_cost = candidate.len_chars
                    best_cov_gain = coverage_delta
                    continue

                if gain > best_gain or (gain == best_gain and candidate.index < usable[best_idx].index):
                    best_idx = idx
                    best_gain = gain
                    best_budget_cost = candidate.len_chars
                    best_cov_gain = coverage_delta

            if best_idx is None:
                break

            # Apply best candidate
            selected_indices.append(best_idx)
            budget_used += best_budget_cost
            selected_text_hashes.add(_sha256(usable[best_idx].text))
            for i in range(n):
                current_cov[i] = max(current_cov[i], similarities[i][best_idx])

            selection_meta.append(
                {
                    "index": usable[best_idx].index,
                    "len": usable[best_idx].len_chars,
                    "unique_ratio": usable[best_idx].unique_char_ratio,
                    "digit_ratio": usable[best_idx].digit_ratio,
                    "punct_ratio": usable[best_idx].punctuation_ratio,
                    "whitespace_ratio": usable[best_idx].whitespace_ratio,
                    "simhash": format(usable[best_idx].simhash, "016x"),
                    "weight": round(usable[best_idx].weight, 4),
                    "coverage_gain": round(best_cov_gain, 4),
                    "score": round(best_gain, 4),
                    "reason": "selected",
                }
            )

            remaining_indices = [i for i in remaining_indices if i != best_idx]

        selected_blocks = [usable[i] for i in selected_indices]
        return selected_blocks, budget_used, selection_meta
