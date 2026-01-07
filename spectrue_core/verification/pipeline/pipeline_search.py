# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

import logging

from spectrue_core.utils.trace import Trace
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.verification.orchestration.orchestrator import ClaimOrchestrator
from spectrue_core.verification.orchestration.phase_runner import PhaseRunner
from spectrue_core.verification.search.search_policy import (
    default_search_policy,
    resolve_profile_name,
)
from spectrue_core.verification.search.search_policy_adapter import (
    apply_search_policy_to_plan,
    apply_claim_retrieval_policy,
    budget_class_for_profile,
    evaluate_locale_decision,
)
from spectrue_core.verification.search.source_utils import canonicalize_sources

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], Awaitable[None]]
CanAddSearch = Callable[[str, str, int | None], bool]


@dataclass(slots=True)
class SearchFlowInput:
    fact: str
    lang: str
    gpt_model: str
    search_type: str
    max_cost: int | None
    article_intent: str
    search_queries: list[str]
    claims: list[dict]
    preloaded_context: str | None
    progress_callback: ProgressCallback | None
    inline_sources: list[dict] = field(default_factory=list)
    # Pipeline profile selection
    pipeline: str | None = None
    """Pipeline profile name (e.g., 'normal', 'deep'). If None, uses search_type mapping."""
    pipeline_overrides: dict | None = None
    """Per-run overrides for pipeline profile settings."""


@dataclass(slots=True)
class SearchFlowState:
    final_context: str
    final_sources: list[dict]
    preloaded_context: str | None
    used_orchestration: bool
    hard_reject: bool = False
    reject_reason: str | None = None
    locale_decisions: dict[str, dict] = field(default_factory=dict)
    execution_state: dict[str, dict] = field(default_factory=dict)


async def run_search_flow(
    *,
    config,
    search_mgr,
    agent,
    can_add_search: CanAddSearch,
    inp: SearchFlowInput,
    state: SearchFlowState,
) -> SearchFlowState:
    """
    Search phase (orchestration or legacy unified + CSE fallback).

    Mutates state by appending context/sources, matching existing pipeline behavior.
    """
    # Feature flags must be explicit booleans; this prevents MagicMock configs
    # (used in tests) from accidentally enabling orchestration.
    features = getattr(getattr(config, "runtime", None), "features", None)
    # NOTE: claim_orchestration flag REMOVED - orchestration is always enabled
    use_orchestration = bool(inp.claims) and not inp.preloaded_context

    # ─────────────────────────────────────────────────────────────────────────
    # Preflight invariant validation for Step-based pipeline
    # ─────────────────────────────────────────────────────────────────────────
    # Run invariant checks before expensive operations (search, LLM calls).
    # For "normal" mode, this validates single-claim and single-language.
    # On violation, logs warning and continues (non-blocking in Phase 5).
    #
    # Future: After Phase 5 stabilizes, this will return early with error.
    # ─────────────────────────────────────────────────────────────────────────
    mode_for_validation = inp.pipeline or (
        "deep" if inp.search_type == "deep" else "normal"
    )

    try:
        from spectrue_core.pipeline import validate_claims_for_mode, PipelineViolation

        await validate_claims_for_mode(mode_for_validation, inp.claims)

        Trace.event(
            "pipeline.preflight_validation.passed",
            {"mode": mode_for_validation, "claims_count": len(inp.claims)},
        )
    except PipelineViolation as pv:
        # M114 Phase 5: Log warning but continue (non-blocking)
        # TODO(Phase6): Make this blocking after stabilization
        logger.warning(
            "Preflight validation failed for mode '%s': %s. Continuing anyway.",
            mode_for_validation,
            pv,
        )
        Trace.event("pipeline.preflight_validation.violation", pv.to_trace_dict())
    except Exception as e:
        # Import or runtime failure — don't block on infra issues
        logger.debug("Preflight validation skipped: %s", e)

    # ─────────────────────────────────────────────────────────────────────────
    # Opt-in full Pipeline.run() execution
    # ─────────────────────────────────────────────────────────────────────────
    # When use_step_pipeline feature flag is True, the entire retrieval
    # is handled by Pipeline.run() instead of PhaseRunner directly.
    # This enables gradual migration to the Step-based architecture.
    # ─────────────────────────────────────────────────────────────────────────
    use_step_pipeline = getattr(features, "use_step_pipeline", False) is True

    if use_step_pipeline and use_orchestration:
        try:
            from spectrue_core.pipeline import execute_pipeline, PipelineViolation

            logger.debug("Using Step-based Pipeline for retrieval")

            result = await execute_pipeline(
                mode_name=mode_for_validation,
                claims=inp.claims,
                search_mgr=search_mgr,
                agent=agent,
                lang=inp.lang,
                search_type=inp.search_type,
                gpt_model=inp.gpt_model,
                max_cost=inp.max_cost,
                inline_sources=inp.inline_sources,
                progress_callback=inp.progress_callback,
            )

            # Transfer results to state
            for src in result.get("sources", []):
                if "url" not in src:
                    continue
                if "domain" not in src:
                    src["domain"] = get_registrable_domain(src.get("url", ""))
                state.final_sources.append(src)
                if src.get("content"):
                    state.final_context += "\n" + src["content"]

            state.execution_state = result.get("execution_state") or {}
            state.used_orchestration = True

            Trace.event(
                "pipeline.step_based.completed",
                {
                    "mode": mode_for_validation,
                    "sources_count": len(result.get("sources", [])),
                },
            )

            return state

        except PipelineViolation as pv:
            logger.error("Pipeline violation in Step-based execution: %s", pv)
            Trace.event("pipeline.step_based.violation", pv.to_trace_dict())
            # Fall through to legacy execution
            use_step_pipeline = False
        except Exception as e:
            logger.warning("Step-based pipeline failed, falling back: %s", e)
            Trace.event("pipeline.step_based.error", {"error": str(e)[:500]})
            # Fall through to legacy execution
            use_step_pipeline = False

    policy = default_search_policy()
    profile_name = resolve_profile_name(inp.search_type)
    profile = policy.get_profile(profile_name)
    search_mgr.set_policy_profile(profile)

    if use_orchestration:
        logger.debug(
            "Using PhaseRunner for progressive widening (orchestration enabled)"
        )

        try:
            # PipelineBuilder removed, always use ClaimOrchestrator for legacy path
            ev_stop_params = None
            orchestrator = ClaimOrchestrator()
            budget_class = budget_class_for_profile(profile)
            execution_plan = orchestrator.build_plan(
                inp.claims, budget_class=budget_class
            )
            execution_plan = apply_search_policy_to_plan(
                execution_plan, profile=profile
            )
            execution_plan = apply_claim_retrieval_policy(
                execution_plan, claims=inp.claims
            )

            locale_config = getattr(getattr(config, "runtime", None), "locale", None)
            default_primary = getattr(locale_config, "default_primary_locale", inp.lang)
            default_fallbacks = getattr(
                locale_config, "default_fallback_locales", ["en"]
            )
            max_fallbacks = getattr(locale_config, "max_fallbacks", 0)

            locale_decisions = {}
            for claim in inp.claims:
                claim_id = str(claim.get("id") or "c1")
                decision = evaluate_locale_decision(
                    claim,
                    profile,
                    default_primary_locale=default_primary,
                    default_fallback_locales=list(default_fallbacks or []),
                    max_fallbacks=int(max_fallbacks)
                    if max_fallbacks is not None
                    else 0,
                )
                locale_decisions[claim_id] = decision
                Trace.event(
                    "search.locale_decision",
                    {
                        "claim_id": claim_id,
                        "primary_locale": decision.primary_locale,
                        "fallback_locales": decision.fallback_locales,
                        "reason_codes": decision.reason_codes,
                    },
                )

            Trace.event(
                "search.policy.applied",
                {
                    "profile": profile.name,
                    "budget_class": execution_plan.budget_class.value,
                    "max_hops": profile.max_hops,
                    "max_results_cap": profile.max_results,
                    "search_depth": profile.search_depth,
                    "channels_allowed": [c.value for c in profile.channels_allowed],
                    "use_policy_by_channel": {
                        k: v.value for k, v in profile.use_policy_by_channel.items()
                    },
                    "quality_thresholds": profile.quality_thresholds.to_dict(),
                    "locale_policy": profile.locale_policy.to_dict(),
                    "stop_conditions": profile.stop_conditions.to_dict(),
                },
            )

            Trace.event(
                "orchestration.plan_built",
                {
                    "claims_count": len(inp.claims),
                    "budget_class": execution_plan.budget_class.value,
                    "phases_per_claim": {
                        c.get("id"): len(execution_plan.get_phases(c.get("id")))
                        for c in inp.claims[:5]
                    },
                },
            )

            runner = PhaseRunner(
                search_mgr,
                progress_callback=inp.progress_callback,
                use_retrieval_loop=True,
                policy_profile=profile,
                can_add_search=can_add_search,
                gpt_model=inp.gpt_model,
                search_type=inp.search_type,
                max_cost=inp.max_cost,
                inline_sources=inp.inline_sources,
                agent=agent,
                ev_stop_params=ev_stop_params,
            )

            phase_evidence = await runner.run_all_claims(inp.claims, execution_plan)

            for claim_id, sources in phase_evidence.items():
                decision = locale_decisions.get(claim_id)
                if decision:
                    phases_completed = runner.execution_state.get_or_create(
                        claim_id
                    ).phases_completed
                    used_locales = []
                    for phase_id in phases_completed:
                        phase = next(
                            (
                                p
                                for p in execution_plan.get_phases(claim_id)
                                if p.phase_id == phase_id
                            ),
                            None,
                        )
                        if phase and phase.locale and phase.locale not in used_locales:
                            used_locales.append(phase.locale)
                    decision.used_locales = (
                        used_locales
                        or decision.used_locales
                        or [decision.primary_locale]
                    )
                    fallback_used = any(
                        loc in decision.fallback_locales for loc in used_locales
                    )
                    decision.sufficiency_triggered = fallback_used
                    if fallback_used:
                        if "fallback_used" not in decision.reason_codes:
                            decision.reason_codes.append("fallback_used")
                    else:
                        if (
                            decision.fallback_locales
                            and "fallback_skipped_sufficient"
                            not in decision.reason_codes
                        ):
                            decision.reason_codes.append("fallback_skipped_sufficient")

                for src in canonicalize_sources(sources):
                    if "url" not in src:
                        continue
                    src["claim_id"] = claim_id
                    if "domain" not in src:
                        src["domain"] = get_registrable_domain(src.get("url", ""))
                    state.final_sources.append(src)
                    if src.get("content"):
                        state.final_context += "\n" + src["content"]

            total_sources = sum(len(s) for s in phase_evidence.values())
            execution_state_payload = runner.execution_state.to_dict()
            execution_state_payload["claim_trace_summaries"] = runner.execution_state.build_trace_summaries(
                plan=execution_plan
            )

            Trace.event(
                "orchestration.completed",
                {
                    "total_sources": total_sources,
                    "claims_with_evidence": len(
                        [c for c in phase_evidence if phase_evidence[c]]
                    ),
                    "execution_state": execution_state_payload,
                },
            )

            state.execution_state = execution_state_payload
            logger.debug(
                "PhaseRunner completed: %d sources from %d claims",
                total_sources,
                len(phase_evidence),
            )
            state.locale_decisions = {
                cid: decision.to_dict() for cid, decision in locale_decisions.items()
            }

        except Exception as e:
            logger.error("CRITICAL: Orchestrator crashed: %s", e, exc_info=True)
            Trace.event("orchestrator.crash", {"error": str(e)[:500]})

            use_orchestration = False

            logger.warning(
                "Safetynet: Orchestrator failed. Switching to SINGLE GENERAL SEARCH (Safety Mode)."
            )

            primary_query = (
                inp.search_queries[0]
                if inp.search_queries
                else (inp.fact[:100] if len(inp.fact) > 100 else inp.fact)
            )

            if primary_query:
                logger.debug("SafetyNet Search: %s", primary_query[:50])
                try:
                    u_ctx, u_srcs = await search_mgr.search_unified(
                        primary_query,
                        topic="general",
                        intent="safetynet",
                    )
                    u_srcs = canonicalize_sources(u_srcs)
                    if u_srcs:
                        state.final_context += "\n" + u_ctx
                        state.final_sources.extend(u_srcs)
                except Exception as se:
                    Trace.event(
                        "pipeline.search.safetynet_error",
                        {"error": str(se)[:200], "query": primary_query[:100]},
                    )
                    logger.warning("[Pipeline] SafetyNet search failed: %s", se)

            state.preloaded_context = state.final_context or " "

    # Legacy search path (when orchestration disabled or failed)
    if not state.preloaded_context and not use_orchestration:
        if inp.progress_callback:
            await inp.progress_callback("searching_unified")

        tavily_topic = "general"

        claims_need_news = any(
            c.get("search_method") == "news"
            for c in inp.claims
            if c.get("importance", 0) >= 0.5
        )

        if claims_need_news or inp.article_intent in ("news", "opinion"):
            tavily_topic = "news"

        primary_query = inp.search_queries[0] if inp.search_queries else ""
        has_results = False

        if primary_query and can_add_search(
            inp.gpt_model, inp.search_type, inp.max_cost
        ):
            current_topic = tavily_topic

            for attempt in range(2):
                logger.debug(
                    "Unified Search (Attempt %d): %s (topic=%s)",
                    attempt + 1,
                    primary_query[:50],
                    current_topic,
                )

                try:
                    u_ctx, u_srcs = await search_mgr.search_unified(
                        primary_query,
                        topic=current_topic,
                        intent=inp.article_intent,
                        article_intent=inp.article_intent,
                    )
                except Exception as se:
                    Trace.event(
                        "pipeline.search.unified_error",
                        {
                            "error": str(se)[:200],
                            "topic": current_topic,
                            "intent": inp.article_intent,
                            "query": primary_query[:100],
                        },
                    )
                    logger.warning("[Pipeline] Unified search failed: %s", se)
                    break

                if not u_srcs:
                    break

                gate_result = await agent.verify_search_relevance(inp.claims, u_srcs)
                is_relevant = gate_result.get("is_relevant")
                gate_status = gate_result.get("status", "RELEVANT")

                # Backward compatibility: explicit boolean rejection should stop the pipeline.
                if is_relevant is False:
                    state.hard_reject = True
                    state.reject_reason = gate_result.get("reason", "irrelevant")
                    state.final_sources = []
                    state.final_context = ""
                    return state

                if gate_status == "RELEVANT":
                    state.final_context += "\n" + u_ctx
                    state.final_sources.extend(canonicalize_sources(u_srcs))
                    has_results = True
                    break

                logger.warning(
                    "Semantic Router: REJECTED results (%s). Reason: %s",
                    gate_status,
                    gate_result.get("reason"),
                )

                if attempt == 0:
                    new_topic = "general" if current_topic == "news" else "news"
                    logger.debug(
                        "Refining Search: Switching topic %s -> %s",
                        current_topic,
                        new_topic,
                    )
                    current_topic = new_topic
                    continue

                logger.warning(
                    "Semantic gating failed after 2 attempts. Retaining %d sources as CONTEXT.",
                    len(u_srcs),
                )
                for src in u_srcs:
                    src["stance"] = "context"
                    src["gating_status"] = gate_status
                    src["gating_reason"] = gate_result.get("reason", "")[:100]
                state.final_context += "\n" + u_ctx
                state.final_sources.extend(canonicalize_sources(u_srcs))
                has_results = True

                Trace.event(
                    "gating.retained_as_context",
                    {
                        "input_sources": len(u_srcs),
                        "match_type": gate_result.get("match_type", gate_status),
                        "reason": gate_result.get("reason", "")[:100],
                        "query": primary_query[:100],
                    },
                )
                break

        # Fallback: Google CSE
        if not has_results and search_mgr.tavily_calls > 0:
            if inp.progress_callback:
                await inp.progress_callback("searching_deep")

            if primary_query:
                try:
                    cse_ctx, cse_srcs = await search_mgr.search_google_cse(
                        primary_query, lang=inp.lang
                    )
                except Exception as ce:
                    Trace.event(
                        "pipeline.search.cse_error",
                        {"error": str(ce)[:200], "query": primary_query[:100]},
                    )
                    logger.warning("[Pipeline] Google CSE search failed: %s", ce)
                    cse_ctx, cse_srcs = ("", [])
                if cse_ctx:
                    state.final_context += "\n\n=== CSE SEARCH ===\n" + cse_ctx
                for res in cse_srcs:
                    url = res.get("link")
                    state.final_sources.append(
                        {
                            "url": url,
                            "domain": get_registrable_domain(url),
                            "title": res.get("title"),
                            "content": res.get("snippet", ""),
                            "source_type": "general",
                            "is_trusted": False,
                        }
                    )
        state.locale_decisions = {
            "c1": {
                "primary_locale": inp.lang,
                "fallback_locales": [],
                "used_locales": [inp.lang],
                "reason_codes": ["legacy_unified_search"],
                "sufficiency_triggered": False,
            }
        }

    state.used_orchestration = use_orchestration
    return state
