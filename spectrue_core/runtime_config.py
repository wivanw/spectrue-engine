# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from spectrue_core.constants import (
    DEFAULT_FALLBACK_LOCALES,
    DEFAULT_LOCALE_MAX_FALLBACKS,
    DEFAULT_PRIMARY_LOCALE,
    DEFAULT_RELATIVE_WINDOW_DAYS,
)

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


def _parse_csv_locales(raw: str) -> list[str]:
    s = (raw or "").strip()
    if not s:
        return []
    parts = [p.strip().lower() for p in re.split(r"[,\n]", s) if p.strip()]
    out: list[str] = []
    for loc in parts:
        if loc and loc not in out:
            out.append(loc)
        if len(out) >= 20:
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

    # Flags
    coverage_chunking: bool = False
    topic_aware_claim_graph: bool = False
    semantic_gating_v2: bool = False
    claim_sanitize: bool = True
    log_redaction: bool = False

    trace_safe_payloads: bool = True

    clean_md_output: bool = True
    # Claim-Centric Orchestration (progressive widening, metadata-driven routing)
    claim_orchestration: bool = True
    # When enabled, the engine may "salvage" a run with missing input text by
    # extracting claims from retrieved web snippets/titles.
    # Default MUST be False: salvage changes the task definition and can hide
    # upstream input failures.
    allow_salvage_mode: bool = False
    # Embeddings for semantic matching
    embeddings_verdict_ready: bool = True  # Use embeddings in verdict_ready_for_claim
    embeddings_clustering: bool = True     # Use embeddings for claim clustering
    embeddings_quotes: bool = True         # Use embeddings for quote extraction


@dataclass(frozen=True)
class EngineDebugFlags:
    engine_debug: bool = False
    log_prompts: bool = False

    trace_max_head_chars: int = 120
    trace_max_inline_chars: int = 600


@dataclass(frozen=True)
class EngineLLMConfig:
    timeout_sec: float = 60.0
    concurrency: int = 6
    nano_timeout_sec: float = 20.0
    nano_max_output_tokens: int = 700  # increased for topics field
    max_output_tokens_general: int = 900
    max_output_tokens_lite: int = 500
    max_output_tokens_deep: int = 1100

    # Responses API configuration
    cluster_timeout_sec: float = 90.0

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
    # Phase runner concurrency limit
    max_concurrent_searches: int = 3


@dataclass(frozen=True)
class CalibrationModelPolicy:
    version: str
    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    fallback_weights: dict[str, float] = field(default_factory=dict)
    fallback_bias: float = 0.0
    score_mode: str = "sigmoid"
    enabled: bool = True


@dataclass(frozen=True)
class CalibrationPolicyConfig:
    claim_utility: CalibrationModelPolicy = field(
        default_factory=lambda: CalibrationModelPolicy(
            version="claim-utility-v1",
            weights={
                "role_weight": 0.35,
                "worthiness": 0.22,
                "harm": 0.22,
                "importance": 0.12,
                "centrality": 0.08,
                "lede_bonus": 0.05,
            },
            fallback_weights={
                "role_weight": 0.35,
                "worthiness": 0.22,
                "harm": 0.22,
                "importance": 0.12,
                "centrality": 0.08,
                "lede_bonus": 0.05,
            },
            score_mode="sigmoid",
        )
    )
    claim_utility_role_weights: dict[str, float] = field(
        default_factory=lambda: {
            "core": 1.0,
            "thesis": 1.0,
            "support": 0.7,
            "counter": 0.6,
            "background": 0.2,
        }
    )
    claim_utility_lede_window: int = 8000
    claim_utility_harm_scale: float = 5.0
    retrieval_confidence: CalibrationModelPolicy = field(
        default_factory=lambda: CalibrationModelPolicy(
            version="retrieval-confidence-v1",
            weights={
                "mean_relevance": 0.50,
                "evidence_likeness": 0.30,
                "source_quality": 0.20,
            },
            fallback_weights={
                "mean_relevance": 0.50,
                "evidence_likeness": 0.30,
                "source_quality": 0.20,
            },
            score_mode="linear",
        )
    )
    retrieval_gain: CalibrationModelPolicy = field(
        default_factory=lambda: CalibrationModelPolicy(
            version="retrieval-gain-v1",
            weights={
                "relevance_gap": 0.30,
                "evidence_gap": 0.25,
                "quality_gap": 0.15,
                "diversity_gap": 0.10,
                "quote_gap": 0.10,
                "conflict_presence": 0.05,
                "temporal_risk": 0.05,
            },
            fallback_weights={
                "relevance_gap": 0.30,
                "evidence_gap": 0.25,
                "quality_gap": 0.15,
                "diversity_gap": 0.10,
                "quote_gap": 0.10,
                "conflict_presence": 0.05,
                "temporal_risk": 0.05,
            },
            score_mode="sigmoid",
        )
    )
    evidence_likeness: CalibrationModelPolicy = field(
        default_factory=lambda: CalibrationModelPolicy(
            version="evidence-likeness-v1",
            weights={
                "has_quote": 1.2,
                "has_chunk": 0.8,
                "has_digits": 0.4,
                "quote_len_norm": 0.6,
                "snippet_len_norm": 0.3,
                "extraction_success": 0.5,
            },
            fallback_weights={
                "has_quote": 1.2,
                "has_chunk": 0.8,
                "has_digits": 0.4,
                "quote_len_norm": 0.6,
                "snippet_len_norm": 0.3,
                "extraction_success": 0.5,
            },
            bias=-1.0,
            fallback_bias=-1.0,
        )
    )
    search_relevance: CalibrationModelPolicy = field(
        default_factory=lambda: CalibrationModelPolicy(
            version="search-relevance-v1",
            weights={
                "lexical_score": 1.4,
                "provider_score": 0.6,
                "snippet_len_norm": 0.4,
                "trusted_domain": 0.5,
                "url_year_freshness": 0.3,
            },
            fallback_weights={
                "lexical_score": 1.4,
                "provider_score": 0.6,
                "snippet_len_norm": 0.4,
                "trusted_domain": 0.5,
                "url_year_freshness": 0.3,
            },
            bias=-0.8,
            fallback_bias=-0.8,
        )
    )

    retrieval_cost_norm: float = 100.0
    retrieval_cost_weight: float = 1.0
    retrieval_min_value_per_cost: float = 0.25
    retrieval_gain_floor: float = 0.15
    retrieval_confidence_low: float = 0.35
    retrieval_confidence_high: float = 0.70

    propagation_min_shrink: float = 0.05
    propagation_max_shrink: float = 0.35
    propagation_similarity_weight: float = 0.7
    propagation_cohesion_weight: float = 0.3

    penalty_conflict_weight: float = 0.10
    penalty_temporal_weight: float = 0.15
    penalty_diversity_weight: float = 0.05


@dataclass(frozen=True)
class TemporalPolicyConfig:
    relative_window_days: int = DEFAULT_RELATIVE_WINDOW_DAYS


@dataclass(frozen=True)
class LocalePolicyConfig:
    default_primary_locale: str = DEFAULT_PRIMARY_LOCALE
    default_fallback_locales: list[str] = field(
        default_factory=lambda: list(DEFAULT_FALLBACK_LOCALES)
    )
    max_fallbacks: int = DEFAULT_LOCALE_MAX_FALLBACKS


@dataclass(frozen=True)
class EngineTunableConfig:
    langdetect_min_prob: float = 0.80
    max_claims_deep: int = 2


@dataclass(frozen=True)
class ContentBudgetConfig:
    max_clean_text_chars_default: int = 120_000
    max_clean_text_chars_huge_input: int = 200_000
    block_min_chars: int = 80
    trace_top_blocks: int = 8
    absolute_guardrail_chars: int = 2_000_000


@dataclass(frozen=True)
class ClaimGraphConfig:
    """
    Hybrid ClaimGraph (B + C) configuration.
    
    Two-stage graph for claim prioritization:
    - B-stage: cheap candidate generation (embeddings + adjacency)
    - C-stage: LLM edge typing (GPT-5 nano)
    """
    # Feature flag
    enabled: bool = True

    # B-Stage parameters
    k_sim: int = 10              # Top-K by embedding similarity
    max_nodes_for_full_pairwise: int = 50  # When to allow full pairwise MST
    edge_pos_gamma: float = 0.6   # Position prior for edge weights (exp decay)

    # Output parameters
    top_k: int = 12              # Key claims to select

    # Priors / PageRank
    pos_prior_gamma: float = 0.12
    w_pos: float = 0.35
    w_supp: float = 0.35
    w_imp: float = 0.2
    w_harm: float = 0.1
    pagerank_alpha: float = 0.85
    pagerank_eps: float = 1e-8
    pagerank_max_iter: int = 200

    # Selection parameters
    selection_budget: float = -1.0  # <=0 means unlimited budget
    default_claim_cost: float = 0.0  # <=0 means do not inject costs
    lambda_rank: float = 0.3
    mu_redundancy: float = 0.1

    # Quality gates
    min_kept_ratio: float = 0.05
    max_kept_ratio: float = 0.60
    trace_top_k: int = 5
    topic_aware: bool = False
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0

    # Layer 2: Structural Claim Prioritization
    structural_prioritization_enabled: bool = True
    structural_weight_threshold: float = 0.5  # min weight for priority boost
    structural_boost: float = 0.1             # importance boost for high structural weight

    # Layer 3: Tension Signal
    tension_signal_enabled: bool = True


    topic_aware: bool = False
    tension_threshold: float = 0.5            # min in_contradict_weight for "high tension"
    tension_boost: float = 0.15               # importance boost for high-tension claims

    # Layer 4: Evidence-Need Routing
    evidence_need_routing_enabled: bool = True


@dataclass(frozen=True)
class EngineRuntimeConfig:
    llm: EngineLLMConfig
    features: EngineFeatureFlags
    debug: EngineDebugFlags
    search: EngineSearchConfig
    calibration: CalibrationPolicyConfig
    temporal: TemporalPolicyConfig
    locale: LocalePolicyConfig
    tunables: EngineTunableConfig
    content_budget: ContentBudgetConfig
    claim_graph: ClaimGraphConfig

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
            trace_max_head_chars=_parse_int(os.getenv("TRACE_MAX_HEAD_CHARS"), default=120, min_v=50, max_v=1000),
            trace_max_inline_chars=_parse_int(os.getenv("TRACE_MAX_INLINE_CHARS"), default=600, min_v=100, max_v=5000),
        )

        features = EngineFeatureFlags(
            query_rewrite_short=_parse_bool(os.getenv("SPECTRUE_LLM_QUERY_REWRITE_SHORT"), default=False),
            trace_enabled=not _parse_bool(os.getenv("SPECTRUE_TRACE_DISABLE"), default=False),
            fulltext_fetch=_parse_bool(os.getenv("SPECTRUE_FULLTEXT_FETCH"), default=False),
            # Feature Flags
            coverage_chunking=_parse_bool(os.getenv("FEATURE_COVERAGE_CHUNKING"), default=False),
            topic_aware_claim_graph=_parse_bool(os.getenv("FEATURE_TOPIC_AWARE_CLAIM_GRAPH"), default=False),
            semantic_gating_v2=_parse_bool(os.getenv("FEATURE_SEMANTIC_GATING_V2"), default=False),
            claim_sanitize=_parse_bool(os.getenv("FEATURE_CLAIM_SANITIZE"), default=True),
            log_redaction=_parse_bool(os.getenv("FEATURE_LOG_REDACTION"), default=False),

            trace_safe_payloads=_parse_bool(os.getenv("TRACE_SAFE_PAYLOADS"), default=True),

            clean_md_output=_parse_bool(os.getenv("FEATURE_CLEAN_MD_OUTPUT"), default=True),
            # Claim Orchestration
            claim_orchestration=_parse_bool(os.getenv("FEATURE_CLAIM_ORCHESTRATION"), default=True),
            # Explicitly gated degraded mode
            allow_salvage_mode=_parse_bool(os.getenv("FEATURE_ALLOW_SALVAGE_MODE"), default=False),
            # Embeddings
            embeddings_verdict_ready=_parse_bool(os.getenv("FEATURE_EMBEDDINGS_VERDICT_READY"), default=True),
            embeddings_clustering=_parse_bool(os.getenv("FEATURE_EMBEDDINGS_CLUSTERING"), default=True),
            embeddings_quotes=_parse_bool(os.getenv("FEATURE_EMBEDDINGS_QUOTES"), default=True),
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
            # Phase runner concurrency limit
            max_concurrent_searches=_parse_int(os.getenv("M80_MAX_CONCURRENT_SEARCHES"), default=3, min_v=1, max_v=10),
        )

        calibration = CalibrationPolicyConfig(
            retrieval_cost_norm=_parse_float(
                os.getenv("CALIBRATION_RETRIEVAL_COST_NORM"), default=100.0, min_v=1.0, max_v=10_000.0
            ),
            retrieval_cost_weight=_parse_float(
                os.getenv("CALIBRATION_RETRIEVAL_COST_WEIGHT"), default=1.0, min_v=0.1, max_v=10.0
            ),
            retrieval_min_value_per_cost=_parse_float(
                os.getenv("CALIBRATION_RETRIEVAL_MIN_VALUE_PER_COST"), default=0.25, min_v=0.0, max_v=10.0
            ),
            retrieval_gain_floor=_parse_float(
                os.getenv("CALIBRATION_RETRIEVAL_GAIN_FLOOR"), default=0.15, min_v=0.0, max_v=1.0
            ),
        )

        temporal = TemporalPolicyConfig(
            relative_window_days=_parse_int(
                os.getenv("SPECTRUE_TEMPORAL_RELATIVE_DAYS"),
                default=DEFAULT_RELATIVE_WINDOW_DAYS,
                min_v=1,
                max_v=3650,
            ),
        )

        fallback_locales = _parse_csv_locales(os.getenv("SPECTRUE_LOCALE_FALLBACKS", ""))
        locale = LocalePolicyConfig(
            default_primary_locale=(os.getenv("SPECTRUE_LOCALE_PRIMARY") or DEFAULT_PRIMARY_LOCALE).strip().lower()
            or DEFAULT_PRIMARY_LOCALE,
            default_fallback_locales=fallback_locales or list(DEFAULT_FALLBACK_LOCALES),
            max_fallbacks=_parse_int(
                os.getenv("SPECTRUE_LOCALE_MAX_FALLBACKS"),
                default=DEFAULT_LOCALE_MAX_FALLBACKS,
                min_v=0,
                max_v=5,
            ),
        )

        tunables = EngineTunableConfig(
            langdetect_min_prob=_parse_float(
                os.getenv("SPECTRUE_LANGDETECT_MIN_PROB"), default=0.80, min_v=0.0, max_v=1.0
            ),
            max_claims_deep=_parse_int(os.getenv("SPECTRUE_MAX_CLAIMS"), default=2, min_v=1, max_v=1000),
        )

        content_budget = ContentBudgetConfig(
            max_clean_text_chars_default=_parse_int(
                os.getenv("CONTENT_BUDGET_MAX_DEFAULT_CHARS"),
                default=120_000,
                min_v=10_000,
                max_v=2_000_000,
            ),
            max_clean_text_chars_huge_input=_parse_int(
                os.getenv("CONTENT_BUDGET_MAX_HUGE_CHARS"),
                default=200_000,
                min_v=50_000,
                max_v=2_000_000,
            ),
            block_min_chars=_parse_int(
                os.getenv("CONTENT_BUDGET_BLOCK_MIN_CHARS"), default=80, min_v=1, max_v=2_000
            ),
            trace_top_blocks=_parse_int(
                os.getenv("CONTENT_BUDGET_TRACE_TOP_BLOCKS"), default=8, min_v=1, max_v=50
            ),
            absolute_guardrail_chars=_parse_int(
                os.getenv("CONTENT_BUDGET_ABSOLUTE_GUARDRAIL"),
                default=2_000_000,
                min_v=100_000,
                max_v=10_000_000,
            ),
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

        # ClaimGraph configuration
        claim_graph = ClaimGraphConfig(
            enabled=_parse_bool(os.getenv("CLAIM_GRAPH_ENABLED"), default=True),
            k_sim=_parse_int(os.getenv("CLAIM_GRAPH_K_SIM"), default=10, min_v=1, max_v=200),
            max_nodes_for_full_pairwise=_parse_int(
                os.getenv("CLAIM_GRAPH_MAX_PAIRWISE"), default=50, min_v=2, max_v=500
            ),
            edge_pos_gamma=_parse_float(os.getenv("CLAIM_GRAPH_EDGE_POS_GAMMA"), default=0.6, min_v=0.05, max_v=10.0),
            top_k=_parse_int(os.getenv("CLAIM_GRAPH_TOP_K"), default=12, min_v=1, max_v=200),
            pos_prior_gamma=_parse_float(os.getenv("CLAIM_GRAPH_POS_GAMMA"), default=0.12, min_v=0.0, max_v=5.0),
            w_pos=_parse_float(os.getenv("CLAIM_GRAPH_W_POS"), default=0.35, min_v=0.0, max_v=5.0),
            w_supp=_parse_float(os.getenv("CLAIM_GRAPH_W_SUPP"), default=0.35, min_v=0.0, max_v=5.0),
            w_imp=_parse_float(os.getenv("CLAIM_GRAPH_W_IMP"), default=0.2, min_v=0.0, max_v=5.0),
            w_harm=_parse_float(os.getenv("CLAIM_GRAPH_W_HARM"), default=0.1, min_v=0.0, max_v=5.0),
            pagerank_alpha=_parse_float(os.getenv("CLAIM_GRAPH_PAGERANK_ALPHA"), default=0.85, min_v=0.0, max_v=1.0),
            pagerank_eps=_parse_float(os.getenv("CLAIM_GRAPH_PAGERANK_EPS"), default=1e-8, min_v=1e-12, max_v=1e-2),
            pagerank_max_iter=_parse_int(os.getenv("CLAIM_GRAPH_PAGERANK_MAX_ITER"), default=200, min_v=10, max_v=5000),
            selection_budget=_parse_float(os.getenv("CLAIM_GRAPH_SELECTION_BUDGET"), default=-1.0, min_v=-1e6, max_v=1e6),
            default_claim_cost=_parse_float(os.getenv("CLAIM_GRAPH_DEFAULT_CLAIM_COST"), default=0.0, min_v=-1e3, max_v=1e3),
            lambda_rank=_parse_float(os.getenv("CLAIM_GRAPH_LAMBDA_RANK"), default=0.3, min_v=0.0, max_v=5.0),
            mu_redundancy=_parse_float(os.getenv("CLAIM_GRAPH_MU_REDUNDANCY"), default=0.1, min_v=0.0, max_v=5.0),
            min_kept_ratio=_parse_float(os.getenv("CLAIM_GRAPH_MIN_KEPT_RATIO"), default=0.05, min_v=0.0, max_v=0.5),
            max_kept_ratio=_parse_float(os.getenv("CLAIM_GRAPH_MAX_KEPT_RATIO"), default=0.60, min_v=0.3, max_v=1.0),
            trace_top_k=_parse_int(os.getenv("CLAIM_GRAPH_TRACE_TOP_K"), default=5, min_v=1, max_v=100),
            beta_prior_alpha=_parse_float(os.getenv("CLAIM_GRAPH_BETA_PRIOR_ALPHA"), default=1.0, min_v=0.1, max_v=10.0),
            beta_prior_beta=_parse_float(os.getenv("CLAIM_GRAPH_BETA_PRIOR_BETA"), default=1.0, min_v=0.1, max_v=10.0),
            # Layer 2-4
            structural_prioritization_enabled=_parse_bool(os.getenv("CLAIM_GRAPH_STRUCTURAL_ENABLED"), default=True),
            structural_weight_threshold=_parse_float(os.getenv("CLAIM_GRAPH_STRUCTURAL_THRESHOLD"), default=0.5, min_v=0.0, max_v=2.0),
            structural_boost=_parse_float(os.getenv("CLAIM_GRAPH_STRUCTURAL_BOOST"), default=0.1, min_v=0.0, max_v=0.5),
            tension_signal_enabled=_parse_bool(os.getenv("CLAIM_GRAPH_TENSION_ENABLED"), default=True),
            tension_threshold=_parse_float(os.getenv("CLAIM_GRAPH_TENSION_THRESHOLD"), default=0.5, min_v=0.0, max_v=2.0),
            tension_boost=_parse_float(os.getenv("CLAIM_GRAPH_TENSION_BOOST"), default=0.15, min_v=0.0, max_v=0.5),
            evidence_need_routing_enabled=_parse_bool(os.getenv("CLAIM_GRAPH_EVIDENCE_NEED_ENABLED"), default=True),
            topic_aware=features.topic_aware_claim_graph,
        )

        return EngineRuntimeConfig(
            llm=llm,
            features=features,
            debug=debug,
            search=search,
            calibration=calibration,
            temporal=temporal,
            locale=locale,
            tunables=tunables,
            content_budget=content_budget,
            claim_graph=claim_graph,
        )

    def to_safe_log_dict(self) -> dict[str, Any]:
        ex = list(self.search.tavily_exclude_domains or [])
        exclude_preview = ex[:3]
        more = max(0, len(ex) - len(exclude_preview))
        return {
            "features": {
                "query_rewrite_short": bool(self.features.query_rewrite_short),
                "trace_enabled": bool(self.features.trace_enabled),
                "fulltext_fetch": bool(self.features.fulltext_fetch),
                "coverage_chunking": bool(self.features.coverage_chunking),
                "topic_aware_graph": bool(self.features.topic_aware_claim_graph),
                "semantic_gating_v2": bool(self.features.semantic_gating_v2),
                "claim_sanitize": bool(self.features.claim_sanitize),
                "trace_safe_payloads": bool(self.features.trace_safe_payloads),
                "clean_md_output": bool(self.features.clean_md_output),
                "claim_orchestration": bool(self.features.claim_orchestration),
                "embeddings_verdict_ready": bool(self.features.embeddings_verdict_ready),
                "embeddings_clustering": bool(self.features.embeddings_clustering),
                "embeddings_quotes": bool(self.features.embeddings_quotes),
            },
            "calibration": {
                "claim_utility_version": self.calibration.claim_utility.version,
                "retrieval_confidence_version": self.calibration.retrieval_confidence.version,
                "retrieval_gain_version": self.calibration.retrieval_gain.version,
                "evidence_likeness_version": self.calibration.evidence_likeness.version,
                "search_relevance_version": self.calibration.search_relevance.version,
                "claim_utility_role_weights": dict(self.calibration.claim_utility_role_weights or {}),
                "claim_utility_lede_window": int(self.calibration.claim_utility_lede_window),
                "claim_utility_harm_scale": float(self.calibration.claim_utility_harm_scale),
                "retrieval_cost_norm": float(self.calibration.retrieval_cost_norm),
                "retrieval_cost_weight": float(self.calibration.retrieval_cost_weight),
                "retrieval_min_value_per_cost": float(self.calibration.retrieval_min_value_per_cost),
                "retrieval_gain_floor": float(self.calibration.retrieval_gain_floor),
                "retrieval_confidence_low": float(self.calibration.retrieval_confidence_low),
                "retrieval_confidence_high": float(self.calibration.retrieval_confidence_high),
            },
            "debug": {
                "engine_debug": bool(self.debug.engine_debug),
                "log_prompts": bool(self.debug.log_prompts),
                "trace_max_head_chars": int(self.debug.trace_max_head_chars),
                "trace_max_inline_chars": int(self.debug.trace_max_inline_chars),
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
            "temporal": {
                "relative_window_days": int(self.temporal.relative_window_days),
            },
            "locale": {
                "default_primary_locale": self.locale.default_primary_locale,
                "default_fallback_locales": list(self.locale.default_fallback_locales),
                "max_fallbacks": int(self.locale.max_fallbacks),
            },
            "tunables": {
                "langdetect_min_prob": float(self.tunables.langdetect_min_prob),
                "max_claims_deep": int(self.tunables.max_claims_deep),
            },
            "content_budget": {
                "max_clean_text_chars_default": int(self.content_budget.max_clean_text_chars_default),
                "max_clean_text_chars_huge_input": int(self.content_budget.max_clean_text_chars_huge_input),
                "block_min_chars": int(self.content_budget.block_min_chars),
                "trace_top_blocks": int(self.content_budget.trace_top_blocks),
                "absolute_guardrail_chars": int(self.content_budget.absolute_guardrail_chars),
            },
            "claim_graph": {
                "enabled": bool(self.claim_graph.enabled),
                "k_sim": int(self.claim_graph.k_sim),
                "max_nodes_for_full_pairwise": int(self.claim_graph.max_nodes_for_full_pairwise),
                "top_k": int(self.claim_graph.top_k),
                "pos_prior_gamma": float(self.claim_graph.pos_prior_gamma),
                "pagerank_alpha": float(self.claim_graph.pagerank_alpha),
                "pagerank_eps": float(self.claim_graph.pagerank_eps),
                "pagerank_max_iter": int(self.claim_graph.pagerank_max_iter),
                "selection_budget": float(self.claim_graph.selection_budget),
                "lambda_rank": float(self.claim_graph.lambda_rank),
                "mu_redundancy": float(self.claim_graph.mu_redundancy),
                "trace_top_k": int(self.claim_graph.trace_top_k),

                "structural_prioritization_enabled": bool(self.claim_graph.structural_prioritization_enabled),
                "tension_signal_enabled": bool(self.claim_graph.tension_signal_enabled),
                "evidence_need_routing_enabled": bool(self.claim_graph.evidence_need_routing_enabled),
            },
        }
