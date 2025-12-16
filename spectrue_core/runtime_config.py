from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional


def _parse_bool(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if not s:
        return default
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_int(raw: Any, *, default: int, min_v: int, max_v: int) -> int:
    try:
        if raw is None:
            v = default
        elif isinstance(raw, int):
            v = raw
        else:
            v = int(str(raw).strip())
    except Exception:
        v = default
    return max(min_v, min(max_v, v))


def _parse_float(raw: Any, *, default: float, min_v: float, max_v: float) -> float:
    try:
        if raw is None:
            v = default
        elif isinstance(raw, (int, float)):
            v = float(raw)
        else:
            v = float(str(raw).strip())
    except Exception:
        v = default
    return max(min_v, min(max_v, v))


def _parse_csv_domains(raw: str) -> list[str]:
    s = (raw or "").strip()
    if not s:
        return []
    parts = [p.strip().lower() for p in re.split(r"[,\n]", s) if p.strip()]
    out: list[str] = []
    for d in parts:
        d = d.lstrip(".")
        if d and d not in out:
            out.append(d)
        if len(out) >= 150:
            break
    return out


@dataclass(frozen=True)
class EngineFeatureFlags:
    # Feature flags: default False unless explicitly enabled.
    query_rewrite_short: bool = False
    # Trace is a local-only debug feature; it is enabled by default and can be disabled via env.
    trace_enabled: bool = True
    # Optional crawler behavior: default False to avoid server-side crawling / IP reputation issues.
    fulltext_fetch: bool = False


@dataclass(frozen=True)
class EngineDebugFlags:
    engine_debug: bool = False
    log_prompts: bool = False


@dataclass(frozen=True)
class EngineLLMConfig:
    timeout_sec: float = 60.0
    concurrency: int = 6
    nano_timeout_sec: float = 20.0
    nano_max_output_tokens: int = 700  # M45: increased for topics field
    max_output_tokens_general: int = 900
    max_output_tokens_lite: int = 500
    max_output_tokens_deep: int = 1100

    # M49: Responses API configuration
    cluster_timeout_sec: float = 30.0



    @property
    def nano_concurrency(self) -> int:
        # Keep nano query generation responsive even when analysis calls are in-flight.
        return max(1, min(16, min(12, int(self.concurrency) * 2)))


@dataclass(frozen=True)
class EngineSearchConfig:
    google_cse_cost: int = 0
    tavily_concurrency: int = 8
    tavily_exclude_domains: list[str] = field(default_factory=list)
    # None means "auto" (depth/domain-filter aware); otherwise forced on/off.
    tavily_include_raw_content: Optional[bool] = None
    tavily_raw_max_results: int = 4


@dataclass(frozen=True)
class EngineTunableConfig:
    langdetect_min_prob: float = 0.80
    max_claims_deep: int = 2


@dataclass(frozen=True)
class EngineRuntimeConfig:
    llm: EngineLLMConfig
    features: EngineFeatureFlags
    debug: EngineDebugFlags
    search: EngineSearchConfig
    tunables: EngineTunableConfig

    @staticmethod
    def load_from_env() -> "EngineRuntimeConfig":
        llm_timeout = _parse_float(os.getenv("OPENAI_TIMEOUT"), default=60.0, min_v=5.0, max_v=300.0)
        llm_conc = _parse_int(os.getenv("OPENAI_CONCURRENCY"), default=6, min_v=1, max_v=16)

        # Query generation (nano) should be tighter than general analysis.
        nano_timeout = _parse_float(os.getenv("SPECTRUE_NANO_TIMEOUT"), default=20.0, min_v=5.0, max_v=30.0)
        nano_max_out = _parse_int(os.getenv("SPECTRUE_NANO_MAX_OUTPUT_TOKENS"), default=700, min_v=120, max_v=1000)

        max_out_general = _parse_int(
            os.getenv("SPECTRUE_LLM_MAX_OUTPUT_TOKENS_GENERAL"), default=900, min_v=200, max_v=4000
        )
        max_out_lite = _parse_int(
            os.getenv("SPECTRUE_LLM_MAX_OUTPUT_TOKENS_LITE"), default=500, min_v=200, max_v=4000
        )
        max_out_deep = _parse_int(
            os.getenv("SPECTRUE_LLM_MAX_OUTPUT_TOKENS_DEEP"), default=1100, min_v=200, max_v=4000
        )


        debug = EngineDebugFlags(
            engine_debug=_parse_bool(os.getenv("SPECTRUE_ENGINE_DEBUG"), default=False),
            log_prompts=_parse_bool(os.getenv("SPECTRUE_ENGINE_LOG_PROMPTS"), default=False),
        )

        features = EngineFeatureFlags(
            query_rewrite_short=_parse_bool(os.getenv("SPECTRUE_LLM_QUERY_REWRITE_SHORT"), default=False),
            trace_enabled=not _parse_bool(os.getenv("SPECTRUE_TRACE_DISABLE"), default=False),
            fulltext_fetch=_parse_bool(os.getenv("SPECTRUE_FULLTEXT_FETCH"), default=False),
        )

        # Search knobs
        raw_include = (os.getenv("SPECTRUE_TAVILY_INCLUDE_RAW_CONTENT") or "auto").strip().lower()
        include_raw: Optional[bool]
        if raw_include in ("1", "true", "yes", "y", "on"):
            include_raw = True
        elif raw_include in ("0", "false", "no", "n", "off"):
            include_raw = False
        else:
            include_raw = None

        search = EngineSearchConfig(
            google_cse_cost=_parse_int(os.getenv("SPECTRUE_GOOGLE_CSE_COST"), default=0, min_v=0, max_v=10_000),
            tavily_concurrency=_parse_int(os.getenv("TAVILY_CONCURRENCY"), default=8, min_v=1, max_v=32),
            tavily_exclude_domains=_parse_csv_domains(os.getenv("SPECTRUE_TAVILY_EXCLUDE_DOMAINS", "")),
            tavily_include_raw_content=include_raw,
            tavily_raw_max_results=_parse_int(os.getenv("SPECTRUE_TAVILY_RAW_MAX_RESULTS"), default=4, min_v=1, max_v=10),
        )

        tunables = EngineTunableConfig(
            langdetect_min_prob=_parse_float(
                os.getenv("SPECTRUE_LANGDETECT_MIN_PROB"), default=0.80, min_v=0.0, max_v=1.0
            ),
            max_claims_deep=_parse_int(os.getenv("SPECTRUE_MAX_CLAIMS"), default=2, min_v=1, max_v=3),
        )

        llm = EngineLLMConfig(
            timeout_sec=float(llm_timeout),
            concurrency=int(llm_conc),
            nano_timeout_sec=float(nano_timeout),
            nano_max_output_tokens=int(nano_max_out),
            max_output_tokens_general=int(max_out_general),
            max_output_tokens_lite=int(max_out_lite),
            max_output_tokens_deep=int(max_out_deep),
        )
        return EngineRuntimeConfig(llm=llm, features=features, debug=debug, search=search, tunables=tunables)

    def to_safe_log_dict(self) -> dict[str, Any]:
        ex = list(self.search.tavily_exclude_domains or [])
        exclude_preview = ex[:3]
        more = max(0, len(ex) - len(exclude_preview))
        return {
            "features": {
                "query_rewrite_short": bool(self.features.query_rewrite_short),
                "trace_enabled": bool(self.features.trace_enabled),
                "fulltext_fetch": bool(self.features.fulltext_fetch),
            },
            "debug": {
                "engine_debug": bool(self.debug.engine_debug),
                "log_prompts": bool(self.debug.log_prompts),
            },
            "llm": {
                "timeout_sec": float(self.llm.timeout_sec),
                "concurrency": int(self.llm.concurrency),
                "nano_timeout_sec": float(self.llm.nano_timeout_sec),
                "nano_max_output_tokens": int(self.llm.nano_max_output_tokens),
                "nano_concurrency": int(self.llm.nano_concurrency),
                "max_output_tokens_general": int(self.llm.max_output_tokens_general),
                "max_output_tokens_lite": int(self.llm.max_output_tokens_lite),
                "max_output_tokens_deep": int(self.llm.max_output_tokens_deep),
                "cluster_timeout_sec": float(self.llm.cluster_timeout_sec),
            },
            "search": {
                "google_cse_cost": int(self.search.google_cse_cost),
                "tavily_concurrency": int(self.search.tavily_concurrency),
                "tavily_exclude_domains_count": len(ex),
                "tavily_exclude_domains_preview": exclude_preview + ([f"...(+{more} more)"] if more else []),
                "tavily_include_raw_content": self.search.tavily_include_raw_content,
                "tavily_raw_max_results": int(self.search.tavily_raw_max_results),
            },
            "tunables": {
                "langdetect_min_prob": float(self.tunables.langdetect_min_prob),
                "max_claims_deep": int(self.tunables.max_claims_deep),
            },
        }

