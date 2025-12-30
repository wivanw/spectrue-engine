from spectrue_core.verification.search_mgr import SearchManager, SEARCH_COSTS
from spectrue_core.verification.evidence import build_evidence_pack
from spectrue_core.verification.source_utils import canonicalize_source, has_evidence_chunk
from spectrue_core.utils.text_processing import clean_article_text, normalize_search_query
from spectrue_core.utils.url_utils import get_registrable_domain
from spectrue_core.utils.trust_utils import enrich_sources_with_trust
from spectrue_core.schema.scoring import BeliefState, ClaimNode, ClaimEdge, ClaimRole, RelationType
from spectrue_core.graph.context import ClaimContextGraph
from spectrue_core.billing.cost_ledger import CostLedger
from spectrue_core import __version__, PROMPT_VERSION, SEARCH_STRATEGY_VERSION
from spectrue_core.billing.estimation import CostEstimator
from spectrue_core.billing.metering import LLMMeter, TavilyMeter
from spectrue_core.billing.progress_emitter import CostProgressEmitter
from spectrue_core.billing.config_loader import load_pricing_policy
from spectrue_core.utils.trace import Trace, current_trace_id
from spectrue_core.config import SpectrueConfig
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.graph import ClaimGraphBuilder
from spectrue_core.verification.pipeline_claim_graph import run_claim_graph_flow
from spectrue_core.verification.pipeline_evidence import EvidenceFlowInput, run_evidence_flow
from spectrue_core.verification.pipeline_oracle import OracleFlowInput, run_oracle_flow
from spectrue_core.verification.pipeline_search import (
    SearchFlowInput,
    SearchFlowState,
    run_search_flow,
)
from spectrue_core.utils.embedding_service import EmbedService
from spectrue_core.verification.claim_dedup import dedup_claims_post_extraction_async
from spectrue_core.verification.calibration_registry import CalibrationRegistry
from spectrue_core.verification.claim_utility import score_claim_utility
from spectrue_core.verification.claim_selection import (
    pick_ui_main_claim,
    top_ui_candidates,
    ui_bucket,
    ui_position,
    ui_score,
    is_admissible_as_main,
)
from spectrue_core.analysis.content_budgeter import ContentBudgeter, TrimResult
from spectrue_core.runtime_config import ContentBudgetConfig
from spectrue_core.verification.pipeline_input import (
    extract_url_anchors,
    is_url_input,
    resolve_url_content,
    restore_urls_from_anchors,
)
from spectrue_core.verification.pipeline_queries import (
    build_assertion_query,
    get_claim_units_for_evidence_mapping,
    get_query_by_role,
    is_fuzzy_duplicate,
    normalize_and_sanitize,
    select_diverse_queries,
    resolve_budgeted_max_queries,
    select_queries_from_claim_units,
)
from spectrue_core.verification.ledger_models import (
    RunLedger,
    PhaseUsage,
    PipelineCounts,
    ReasonCode as LedgerReasonCode,
    ReasonSummary,
    ClaimLedgerEntry,
    ClusterLedgerEntry,
    BudgetAllocation,
    RetrievalEvaluation,
)
from spectrue_core.verification.reason_codes import ReasonCodes
from spectrue_core.verification.search_policy import decide_claim_policy
from spectrue_core.verification.execution_plan import PolicyMode
from spectrue_core.verification.target_selection import (
    select_verification_targets,
    propagate_deferred_verdicts,
)
from spectrue_core.verification.costs import summarize_reason_codes, map_stage_costs_to_phases
import logging
import asyncio
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

# Intents that should trigger Oracle check
ORACLE_CHECK_INTENTS = {"news", "evergreen", "official"}
# Intents that should skip Oracle (opinion, prediction)
ORACLE_SKIP_INTENTS = {"opinion", "prediction"}


@dataclass(slots=True)
class _PreparedInput:
    fact: str
    original_fact: str
    final_context: str
    final_sources: list
    inline_sources: list[dict]


class ValidationPipeline:
    """
    Orchestrates the fact-checking waterfall process.
    """
    def __init__(self, config: SpectrueConfig, agent: FactCheckerAgent, translation_service=None):
        self.config = config
        self.agent = agent
        self._calibration_registry = CalibrationRegistry.from_runtime(
            getattr(config, "runtime", None)
        )
        EmbedService.configure(openai_api_key=getattr(config, "openai_api_key", None))
        # Pass oracle_skill to SearchManager for hybrid mode
        self.search_mgr = SearchManager(config, oracle_validator=agent.oracle_skill)
        # Optional translation service for Oracle result localization
        self.translation_service = translation_service
        
        # ClaimGraph for key claim identification
        self._claim_graph: ClaimGraphBuilder | None = None
        claim_graph_enabled = (
            getattr(getattr(getattr(config, "runtime", None), "claim_graph", None), "enabled", False)
            is True
        )
        if config and claim_graph_enabled:
            self._claim_graph = ClaimGraphBuilder(
                config=config.runtime.claim_graph,
                edge_typing_skill=agent.edge_typing_skill,
            )

        self._content_budget_config = (
            getattr(getattr(config, "runtime", None), "content_budget", None)
        )
        if not isinstance(self._content_budget_config, ContentBudgetConfig):
            self._content_budget_config = ContentBudgetConfig()
        self._claim_extraction_text: str = ""

    def _apply_content_budget(self, text: str, *, source: str) -> tuple[str, TrimResult | None]:
        """
        Apply deterministic content budgeting to plain text before LLM steps.
        """
        if not text:
            return text, None
        cfg = self._content_budget_config
        raw_len = len(text)
        if raw_len > int(cfg.absolute_guardrail_chars):
            Trace.event(
                "content_budgeter.guardrail",
                {"raw_len": raw_len, "absolute_guardrail_chars": int(cfg.absolute_guardrail_chars), "source": source},
            )
            raise ValueError("Input too large to process safely")

        if raw_len <= int(cfg.max_clean_text_chars_default):
            return text, None

        budgeter = ContentBudgeter(cfg)
        result = budgeter.trim(text)
        Trace.event("content_budgeter.blocks", result.trace_blocks_payload())
        Trace.event(
            "content_budgeter.selection",
            result.trace_selection_payload(getattr(cfg, "trace_top_blocks", 8)),
        )
        return result.trimmed_text, result

    def _trace_input_summary(
        self, *, source: str, raw_text: str, cleaned_text: str, budget_result: TrimResult | None
    ) -> None:
        raw_sha = budget_result.raw_sha256 if budget_result else hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        cleaned_sha = (
            budget_result.trimmed_sha256 if budget_result else hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        )
        Trace.event(
            "analysis.input_summary",
            {
                "source": source,
                "raw_len": len(raw_text),
                "cleaned_len": len(cleaned_text),
                "raw_sha256": raw_sha,
                "cleaned_sha256": cleaned_sha,
                "budget_applied": bool(budget_result),
            },
        )

    async def execute(
        self,
        fact: str,
        search_type: str,
        gpt_model: str,
        lang: str,
        content_lang: str | None = None,
        max_cost: int | None = None,
        progress_callback=None,
        preloaded_context: str | None = None,
        preloaded_sources: list | None = None,
        needs_cleaning: bool = False,  # Text from extension needs LLM cleaning
        source_url: str | None = None,  # Original URL for inline source exclusion
        extract_claims_only: bool = False,  # M105: Deep mode - just extract claims, skip verification
    ) -> dict:
        Trace.event(
            "pipeline.run.start",
            {
                "search_type": search_type,
                "lang": lang,
                "needs_cleaning": needs_cleaning,
                "has_preloaded_context": bool(preloaded_context),
                "has_preloaded_sources": bool(preloaded_sources),
                "extract_claims_only": extract_claims_only,
            },
        )

        policy = load_pricing_policy()
        ledger = CostLedger(run_id=current_trace_id())
        tavily_meter = TavilyMeter(ledger=ledger, policy=policy)
        llm_meter = LLMMeter(ledger=ledger, policy=policy)
        progress_emitter = CostProgressEmitter(
            ledger=ledger,
            min_delta_to_show=policy.min_delta_to_show,
            emit_cost_deltas=policy.emit_cost_deltas,
        )

        run_ledger = RunLedger(
            run_id=current_trace_id(),
            engine_version=__version__,
            prompt_version=PROMPT_VERSION,
            search_strategy=SEARCH_STRATEGY_VERSION,
            counts=PipelineCounts(),
        )
        phase_timings: dict[str, int] = {}
        phase_reason_codes: dict[str, list[LedgerReasonCode]] = {}
        reason_events: list[dict] = []
        claim_entries: dict[str, ClaimLedgerEntry] = {}

        def _start_phase(phase: str) -> None:
            Trace.phase_start(phase)

        def _end_phase(phase: str) -> None:
            duration = Trace.phase_end(phase)
            if duration is not None:
                phase_timings[phase] = int(duration)

        def _record_reason(
            spec,
            *,
            claim_id: str | None = None,
            count: int = 1,
            sc_cost: float = 0.0,
            tc_cost: float = 0.0,
        ) -> LedgerReasonCode:
            code = spec.qualified()
            entry = LedgerReasonCode(
                code=code,
                label=spec.label,
                phase=spec.phase,
                action=spec.action,
                count=count,
            )
            phase_reason_codes.setdefault(spec.phase, []).append(entry)
            reason_events.append(
                {
                    "code": code,
                    "label": spec.label,
                    "phase": spec.phase,
                    "action": spec.action,
                    "count": count,
                    "sc_cost": sc_cost,
                    "tc_cost": tc_cost,
                }
            )
            Trace.reason_code(
                code=code,
                phase=spec.phase,
                action=spec.action,
                label=spec.label,
                claim_id=claim_id,
            )
            return entry

        def _budget_allocation_for_metadata(metadata) -> BudgetAllocation:
            worthiness = float(getattr(metadata, "check_worthiness", 0.5) or 0.5)
            if worthiness >= 0.75:
                return BudgetAllocation(
                    worthiness_tier="high",
                    max_queries=3,
                    max_docs=8,
                    max_escalations=2,
                    defer_allowed=False,
                )
            if worthiness <= 0.35:
                return BudgetAllocation(
                    worthiness_tier="low",
                    max_queries=1,
                    max_docs=3,
                    max_escalations=0,
                    defer_allowed=True,
                )
            return BudgetAllocation(
                worthiness_tier="medium",
                max_queries=2,
                max_docs=5,
                max_escalations=1,
                defer_allowed=False,
            )

        prior_llm_meter = getattr(self.agent.llm_client, "_meter", None)
        prior_tavily_meter = getattr(self.search_mgr.web_tool._tavily, "_meter", None)
        self.agent.llm_client._meter = llm_meter
        self.search_mgr.web_tool._tavily._meter = tavily_meter

        async def _progress(stage: str, processed: int | None = None, total: int | None = None) -> None:
            if progress_callback:
                await progress_callback(stage, processed, total)
            payload = progress_emitter.maybe_emit(stage=stage)
            if payload:
                Trace.progress_cost_delta(
                    stage=payload.stage,
                    delta=payload.delta,
                    total=payload.total,
                )

        def _attach_cost_summary(payload: dict) -> dict:
            summary_obj = ledger.get_summary()
            phase_costs = map_stage_costs_to_phases(summary_obj.by_stage_credits)
            phase_order = [
                "extraction",
                "graph",
                "query_build",
                "retrieval",
                "evidence_eval",
                "verdict",
            ]
            run_ledger.phase_usage = [
                PhaseUsage(
                    phase=phase,
                    sc_cost=float(phase_costs.get(phase, 0.0)),
                    tc_cost=0.0,
                    duration_ms=int(phase_timings.get(phase, 0)),
                    reason_codes=phase_reason_codes.get(phase, []),
                )
                for phase in phase_order
            ]
            run_ledger.top_reason_codes = [
                ReasonSummary(**item) for item in summarize_reason_codes(reason_events)
            ]
            run_ledger.claim_entries = list(claim_entries.values())
            run_ledger.counts.llm_calls_total = len(
                [e for e in summary_obj.events if e.provider == "openai"]
            )

            ledger.set_phase_usage([pu.to_dict() for pu in run_ledger.phase_usage])
            ledger.set_reason_summaries([rs.to_dict() for rs in run_ledger.top_reason_codes])

            summary = ledger.get_summary().to_dict()
            payload["cost_summary"] = summary
            audit = payload.get("audit") or {}
            audit["usage_ledger"] = run_ledger.to_dict()
            payload["audit"] = audit
            Trace.event(
                "usage_ledger.summary",
                {
                    "counts": run_ledger.counts.to_dict(),
                    "phase_usage": [
                        {
                            "phase": pu.phase,
                            "sc_cost": pu.sc_cost,
                            "tc_cost": pu.tc_cost,
                            "duration_ms": pu.duration_ms,
                        }
                        for pu in run_ledger.phase_usage
                    ],
                    "top_reason_codes": [
                        rc.to_dict() for rc in run_ledger.top_reason_codes
                    ],
                },
            )
            Trace.event("cost_summary.attached", {
                "total_credits": summary.get("total_credits"),
                "total_usd": summary.get("total_usd"),
                "event_count": len(summary.get("events", [])),
            })
            return payload

        await _progress("analyzing_input")

        self.search_mgr.reset_metrics()

        try:
            _start_phase("extraction")
            prepared = await self._prepare_input(
                fact=fact,
                preloaded_context=preloaded_context,
                preloaded_sources=preloaded_sources,
                needs_cleaning=needs_cleaning,
                source_url=source_url,
                progress_callback=_progress,
            )
            fact_before_budget = prepared.fact
            original_fact = prepared.original_fact
            final_context = prepared.final_context
            final_sources = prepared.final_sources
            inline_sources = prepared.inline_sources

            budget_result: TrimResult | None = None
            if not self._is_url(fact_before_budget):
                fact, budget_result = self._apply_content_budget(fact_before_budget, source="plain")
            else:
                fact = fact_before_budget

            source_label = "url" if self._is_url(original_fact) else ("clean_plain" if needs_cleaning else "plain")
            self._trace_input_summary(
                source=source_label,
                raw_text=fact_before_budget,
                cleaned_text=fact,
                budget_result=budget_result,
            )

            claims, should_check_oracle, article_intent, fast_query, stitched_claim_text = await self._extract_claims(
                fact=fact,
                lang=lang,
                progress_callback=_progress,
            )
            _end_phase("extraction")
            self._claim_extraction_text = stitched_claim_text
            anchor_claim = None
            anchor_claim_id = None

            # Semantic claim dedup right after extraction (pre-oracle/graph/search).
            # This reduces cost + prevents anchor/secondary duplicates.
            try:
                before_n = len(claims or [])
                claims, dedup_pairs = await dedup_claims_post_extraction_async(claims or [], tau=0.90)
                after_n = len(claims or [])
                if dedup_pairs:
                    Trace.event(
                        "claims.dedup_post_extraction",
                        {
                            "before": before_n,
                            "after": after_n,
                            "removed": max(before_n - after_n, 0),
                            "tau": 0.90,
                            "pairs": [
                                {
                                    "canonical_id": p.canonical_id,
                                    "duplicate_id": p.duplicate_id,
                                    "sim": p.similarity,
                                }
                                for p in dedup_pairs[:50]  # trace safety cap
                            ],
                        },
                    )
                else:
                    Trace.event(
                        "claims.dedup_post_extraction",
                        {"before": before_n, "after": after_n, "removed": 0, "tau": 0.90, "pairs": []},
                    )
            except Exception as e:
                # Non-fatal: if embeddings unavailable or error occurs, proceed with raw claims.
                Trace.event("claims.dedup_post_extraction.failed", {"error": str(e)})

            # M105: Deep mode - just extract claims, skip full verification
            if extract_claims_only:
                Trace.event("pipeline.extract_claims_only.returning", {"claims_count": len(claims or [])})
                # Note: extraction phase already ended at line 416
                return _attach_cost_summary({
                    "verified_score": 0.0,
                    "text": fact,
                    "sources": final_sources,
                    "details": [],
                    "_extracted_claims": claims,  # Pass claims back to engine for per-claim verification
                    "cost": ledger.total_credits,
                })

            final_sources = await self._verify_inline_sources(
                inline_sources=inline_sources,
                claims=claims,
                fact=fact,
                final_sources=final_sources,
                progress_callback=_progress,
            )

            oracle_flow = await run_oracle_flow(
                self.search_mgr,
                inp=OracleFlowInput(
                    original_fact=original_fact,
                    fast_query=fast_query,
                    lang=lang,
                    article_intent=article_intent,
                    should_check_oracle=should_check_oracle,
                    claims=claims,
                    oracle_check_intents=ORACLE_CHECK_INTENTS,
                    oracle_skip_intents=ORACLE_SKIP_INTENTS,
                    progress_callback=_progress,
                ),
                finalize_jackpot=lambda oracle_result: self._finalize_oracle_hybrid(
                    oracle_result,
                    original_fact,
                    lang=lang,
                    progress_callback=_progress,
                ),
                create_evidence_source=self._create_oracle_source,
            )

            if oracle_flow.early_result:
                Trace.event("pipeline.run.completed", {"outcome": "oracle_early"})
                return _attach_cost_summary(oracle_flow.early_result)

            if oracle_flow.evidence_source:
                canonical = canonicalize_source(oracle_flow.evidence_source)
                final_sources.append(canonical or oracle_flow.evidence_source)

            _start_phase("graph")
            graph_flow_result = await run_claim_graph_flow(
                self._claim_graph,
                claims=claims,
                runtime_config=self.config.runtime,
                progress_callback=_progress,
            )

            if claims:
                centrality_map: dict[str, float] = {}
                gr_obj = getattr(graph_flow_result, "graph_result", None) if graph_flow_result else None
                ranked_all = getattr(gr_obj, "all_ranked", None) or []
                max_c = max((float(getattr(r, "centrality_score", 0.0) or 0.0) for r in ranked_all), default=1.0)
                for r in ranked_all:
                    try:
                        centrality_map[r.claim_id] = float(getattr(r, "centrality_score", 0.0) or 0.0) / max_c
                    except Exception:
                        continue
                max_pos = max((ui_position(c) for c in claims if isinstance(c, dict)), default=0)
                ui_anchor_claim = pick_ui_main_claim(
                    claims,
                    calibration_registry=self._calibration_registry,
                    centrality_map=centrality_map,
                )
                anchor_claim = ui_anchor_claim or (claims[0] if claims else None)
                anchor_claim_id = anchor_claim.get("id") or anchor_claim.get("claim_id")
                ui_candidates = top_ui_candidates(
                    claims,
                    limit=3,
                    calibration_registry=self._calibration_registry,
                    centrality_map=centrality_map,
                )
                def _role_val(c: dict) -> str | None:
                    val = c.get("claim_role") if isinstance(c, dict) else None
                    return str(val) if val is not None else None
                anchor_score, anchor_trace = score_claim_utility(
                    anchor_claim,
                    centrality_map=centrality_map,
                    max_pos=max_pos,
                    calibration_registry=self._calibration_registry,
                )
                Trace.event(
                    "anchor_selection.ui_main",
                    {
                        "ui_anchor_id": anchor_claim_id,
                        "ui_anchor_role": _role_val(anchor_claim),
                        "ui_anchor_kind": anchor_claim.get("type"),
                        "ui_anchor_pos": ui_position(anchor_claim),
                        "ui_anchor_worthiness": anchor_claim.get("check_worthiness"),
                        "ui_anchor_harm": anchor_claim.get("harm_potential"),
                        "utility_anchor_score": anchor_score,
                        "utility_anchor_trace": anchor_trace,
                        "candidates": [
                            {
                                "id": c.get("id") or c.get("claim_id"),
                                "role": _role_val(c),
                                "kind": c.get("type"),
                                "bucket": ui_bucket(_role_val(c)),
                                "utility_score": ui_score(
                                    c,
                                    calibration_registry=self._calibration_registry,
                                    centrality_map=centrality_map,
                                    max_pos=max_pos,
                                ),
                                "pos": ui_position(c),
                            }
                            for c in ui_candidates
                        ],
                        "admissible_count": len([c for c in claims if is_admissible_as_main(c)]),
                        "fallback_used": ui_anchor_claim is None,
                    },
                )

            # Build Context Graph for Belief Propagation
            context_graph = None
            if graph_flow_result and graph_flow_result.graph_result and not graph_flow_result.graph_result.disabled:
                context_graph = ClaimContextGraph()
                gr = graph_flow_result.graph_result
                
                # Add nodes
                for c in claims:
                    role_str = c.get("type", "support").lower()
                    # Map simplified type to ClaimRole
                    role = ClaimRole.SUPPORT
                    if role_str == "core":
                        role = ClaimRole.THESIS
                    elif role_str == "background":
                        role = ClaimRole.BACKGROUND
                    elif role_str == "counter":
                        role = ClaimRole.COUNTER
                    
                    node = ClaimNode(
                        claim_id=c.get("id"),
                        text=c.get("text", ""),
                        role=role
                    )
                    context_graph.add_node(node)
                
                # Add edges
                if getattr(gr, "typed_edges", None):
                    for te in gr.typed_edges:
                        rel_val = te.relation.value.lower()
                        rel_type = RelationType.SUPPORTS
                        if "contradict" in rel_val or "conflict" in rel_val:
                            rel_type = RelationType.CONTRADICTS
                        elif "entail" in rel_val:
                            rel_type = RelationType.ENTAILS
                        
                        edge = ClaimEdge(
                            source_id=te.src_id,
                            target_id=te.dst_id,
                            relation=rel_type,
                            weight=te.score
                        )
                        context_graph.add_edge(edge)
            _end_phase("graph")

            _start_phase("query_build")
            eligible_claims: list[dict] = []
            for claim in claims:
                claim_id = str(claim.get("id") or "c1")
                metadata = claim.get("metadata")
                decision = decide_claim_policy(metadata)
                policy_mode = decision.mode.value
                claim["policy_mode"] = policy_mode
                budget = _budget_allocation_for_metadata(metadata)
                claim["budget_allocation"] = budget.to_dict()

                policy_reasons: list[LedgerReasonCode] = []
                if decision.mode == PolicyMode.SKIP:
                    policy_reasons.append(_record_reason(ReasonCodes.POLICY_SKIP, claim_id=claim_id))
                elif decision.mode == PolicyMode.CHEAP:
                    policy_reasons.append(_record_reason(ReasonCodes.POLICY_CHEAP, claim_id=claim_id))
                else:
                    policy_reasons.append(_record_reason(ReasonCodes.POLICY_FULL, claim_id=claim_id))

                if budget.worthiness_tier == "low":
                    policy_reasons.append(_record_reason(ReasonCodes.BUDGET_LOW, claim_id=claim_id))
                elif budget.worthiness_tier == "high":
                    policy_reasons.append(_record_reason(ReasonCodes.BUDGET_HIGH, claim_id=claim_id))
                else:
                    policy_reasons.append(_record_reason(ReasonCodes.BUDGET_MEDIUM, claim_id=claim_id))

                claim_entry = ClaimLedgerEntry(
                    claim_id=claim_id,
                    policy_mode=policy_mode,
                    policy_reasons=policy_reasons,
                    budget_allocation=budget,
                )
                claim_entries[claim_id] = claim_entry

                if decision.mode != PolicyMode.SKIP:
                    eligible_claims.append(claim)

            run_ledger.counts.claims_total = len(claims)
            run_ledger.counts.claims_eligible = len(eligible_claims)

            cluster_map: dict[str, list[str]] = {}
            for claim in eligible_claims:
                cluster_id = claim.get("cluster_id") or claim.get("topic_key") or "cluster_default"
                cluster_map.setdefault(str(cluster_id), []).append(str(claim.get("id") or "c1"))

            run_ledger.cluster_entries = [
                ClusterLedgerEntry(
                    cluster_id=cluster_id,
                    claim_ids=claim_ids,
                    shared_query_ids=[],
                    budget_allocation=None,
                    duplicate_query_savings=0,
                )
                for cluster_id, claim_ids in cluster_map.items()
            ]

            search_queries = self._select_diverse_queries(eligible_claims, max_queries=3, fact_fallback=fact)
            
            # TARGET SELECTION GATE: Only top-K claims get actual Tavily searches
            # This prevents per-claim search explosion (6 claims × 2 searches = 12 Tavily calls)
            # Instead, we search for top 2 key claims and share evidence with others
            budget_class_str = "minimal"  # Default to most cost-effective
            if search_type == "smart":
                budget_class_str = "standard"
            elif search_type == "deep":
                budget_class_str = "deep"
            
            graph_result_for_selection = None
            if graph_flow_result and graph_flow_result.graph_result:
                graph_result_for_selection = graph_flow_result.graph_result
            
            target_selection = select_verification_targets(
                claims=eligible_claims,
                max_targets=2,  # Core limit: max 2 claims trigger Tavily
                graph_result=graph_result_for_selection,
                budget_class=budget_class_str,
            )
            
            # Only target claims go through retrieval
            retrieval_claims = target_selection.targets
            deferred_claims = target_selection.deferred
            
            # Mark deferred claims in ledger
            for claim in deferred_claims:
                claim_id = str(claim.get("id") or "c1")
                reason = target_selection.reasons.get(claim_id, "deferred_no_search")
                entry = claim_entries.get(claim_id)
                if entry:
                    entry.policy_reasons.append(
                        _record_reason(ReasonCodes.CLAIM_DEFERRED, claim_id=claim_id)
                    )
                claim["deferred_from_search"] = True
                claim["deferred_reason"] = reason
            
            _end_phase("query_build")

            _start_phase("retrieval")
            search_state = await run_search_flow(
                config=self.config,
                search_mgr=self.search_mgr,
                agent=self.agent,
                can_add_search=self._can_add_search,
                inp=SearchFlowInput(
                    fact=fact,
                    lang=lang,
                    gpt_model=gpt_model,
                    search_type=search_type,
                    max_cost=max_cost,
                    article_intent=article_intent,
                    search_queries=search_queries,
                    claims=retrieval_claims,  # Only targets, not all eligible!
                    preloaded_context=preloaded_context,
                    progress_callback=_progress,
                    inline_sources=prepared.inline_sources,  # Pass verified inline sources
                ),
                state=SearchFlowState(
                    final_context=final_context,
                    final_sources=final_sources,
                    preloaded_context=preloaded_context,
                    used_orchestration=False,
                ),
            )
            _end_phase("retrieval")
            final_context = search_state.final_context
            final_sources = search_state.final_sources

            run_ledger.counts.docs_total = len(final_sources)
            run_ledger.counts.evidence_units_total = len(
                [s for s in final_sources if has_evidence_chunk(s)]
            )

            if search_state.execution_state:
                query_total = 0
                for claim_id, state in search_state.execution_state.items():
                    hops = state.get("hops", []) if isinstance(state, dict) else []
                    query_total += len(hops)
                    entry = claim_entries.get(str(claim_id))
                    if entry:
                        entry.queries_used = len(hops)
                        entry.docs_used = sum(
                            int(h.get("results_count", 0) or 0) for h in hops
                        )
                        if state.get("is_sufficient"):
                            sufficiency_reason = str(state.get("sufficiency_reason") or "")
                            if sufficiency_reason != "confidence_high":
                                entry.stop_reason = _record_reason(
                                    ReasonCodes.EVIDENCE_SUFFICIENT,
                                    claim_id=str(claim_id),
                                )
                        else:
                            stop_reason = state.get("stop_reason")
                            if stop_reason == "max_hops_reached":
                                entry.stop_reason = _record_reason(
                                    ReasonCodes.RETRIEVAL_STOP_MAX_HOPS,
                                    claim_id=str(claim_id),
                                )
                            elif stop_reason == "followup_failed":
                                entry.stop_reason = _record_reason(
                                    ReasonCodes.RETRIEVAL_STOP_FOLLOWUP_FAILED,
                                    claim_id=str(claim_id),
                                )
                        registry = getattr(self, "_calibration_registry", None)
                        if registry is not None:
                            low_threshold = float(getattr(registry.policy, "retrieval_confidence_low", 0.35) or 0.35)
                            high_threshold = float(getattr(registry.policy, "retrieval_confidence_high", 0.70) or 0.70)
                        else:
                            low_threshold = 0.35
                            high_threshold = 0.70
                        if high_threshold <= low_threshold:
                            high_threshold = min(1.0, low_threshold + 0.05)
                        for hop in hops:
                            eval_data = hop.get("retrieval_eval", {}) if isinstance(hop, dict) else {}
                            action = eval_data.get("action", "continue")
                            reason_spec = ReasonCodes.RETRIEVAL_CONF_MED
                            if action == "stop_early":
                                reason_spec = ReasonCodes.RETRIEVAL_STOP_EARLY
                            elif action in ("refine_query", "change_language", "restrict_domains", "change_channel"):
                                reason_spec = ReasonCodes.RETRIEVAL_CORRECTION
                            elif eval_data.get("retrieval_confidence", 0.0) <= low_threshold:
                                reason_spec = ReasonCodes.RETRIEVAL_CONF_LOW
                            elif eval_data.get("retrieval_confidence", 0.0) >= high_threshold:
                                reason_spec = ReasonCodes.RETRIEVAL_CONF_HIGH

                            reason_code = _record_reason(reason_spec, claim_id=str(claim_id))
                            entry.retrieval_evaluations.append(
                                RetrievalEvaluation(
                                    cycle_index=int(hop.get("hop_index", 0)),
                                    relevance_score=float(eval_data.get("relevance_score", 0.0)),
                                    evidence_likeness_score=float(eval_data.get("evidence_likeness_score", 0.0)),
                                    source_quality_score=float(eval_data.get("source_quality_score", 0.0)),
                                    retrieval_confidence=float(eval_data.get("retrieval_confidence", 0.0)),
                                    action=str(action),
                                    reason_code=reason_code,
                                    expected_gain=float(eval_data.get("expected_gain", 0.0)),
                                    expected_cost=float(eval_data.get("expected_cost", 0.0)),
                                    value_per_cost=float(eval_data.get("value_per_cost", 0.0)),
                                )
                            )
                run_ledger.counts.queries_total = query_total
            else:
                run_ledger.counts.queries_total = len(search_queries)

            if getattr(search_state, "hard_reject", False):
                reason = getattr(search_state, "reject_reason", "irrelevant")
                
                # M105: Fallback Extraction Check
                # If rejection was due to missing input data (e.g. initial URL fetch failed)
                # but we successfully retrieved search results, try to extract claims now.
                claims_recovered = False
                if reason == "No input data" and final_sources:
                    Trace.event("pipeline.fallback_extraction.start", {"sources_count": len(final_sources)})
                    try:
                        # Synthesize text from top results
                        synth_text = "\n\n".join([
                            f"{s.get('title', '')}\n{s.get('content', '') or s.get('snippet', '')}"
                            for s in final_sources[:3]
                        ])
                        
                        fallback_claims, _, _, _ = await self.agent.claims_skill.extract_claims(
                            synth_text[:15000],  # Cap context
                            lang=lang,
                            max_claims=3
                        )
                        
                        if fallback_claims:
                            claims = fallback_claims
                            retrieval_claims = fallback_claims
                            # Assume these are now the 'retrieved' claims for downstream logic
                            claims_recovered = True
                            Trace.event("pipeline.fallback_extraction.success", {"claims_count": len(fallback_claims)})
                            
                            # Register new claims in ledger
                            for c in fallback_claims:
                                cid = c.get("id")
                                if cid:
                                    claim_entries[cid] = ClaimLedgerEntry(
                                        claim_id=cid,
                                        policy_mode="fallback",
                                    )
                    except Exception as e:
                        Trace.event("pipeline.fallback_extraction.failed", {"error": str(e)})

                if not claims_recovered:
                    Trace.event("pipeline.run.completed", {"outcome": "hard_reject", "reason": reason})
                    return _attach_cost_summary({
                        "verified_score": 0.0,
                        "analysis": f"Search results are irrelevant to the claim: {reason}",
                        "rationale": reason,
                        "cost": self.search_mgr.calculate_cost(gpt_model, search_type),
                        "text": fact,
                        "search_meta": self.search_mgr.get_search_meta(),
                        "sources": [],
                        "details": [],
                    })

            # Neutral prior (tier does not influence veracity)
            prior_log_odds = 0.0
            
            prior_belief = BeliefState(log_odds=prior_log_odds)

            _start_phase("evidence_eval")
            Trace.event(
                "evidence_flow.claim_scope",
                {
                    "claims_total": len(claims or []),
                    "retrieval_claims": len(retrieval_claims or []),
                    "deferred_claims": len(deferred_claims or []),
                    "retrieval_ids": [c.get("id") for c in (retrieval_claims or [])],
                    "deferred_ids": [c.get("id") for c in (deferred_claims or [])],
                    "sources_total": len(final_sources or []),
                },
            )
            claims_for_scoring = retrieval_claims if retrieval_claims else claims
            result = await run_evidence_flow(
                agent=self.agent,
                search_mgr=self.search_mgr,
                build_evidence_pack=build_evidence_pack,
                enrich_sources_with_trust=enrich_sources_with_trust,
                calibration_registry=self._calibration_registry,
                inp=EvidenceFlowInput(
                    fact=fact,
                    original_fact=original_fact,
                    lang=lang,
                    content_lang=content_lang,
                    gpt_model=gpt_model,
                    search_type=search_type,
                    progress_callback=_progress,
                    prior_belief=prior_belief,
                    context_graph=context_graph,
                    claim_extraction_text=self._claim_extraction_text,
                ),
                claims=claims_for_scoring,
                sources=final_sources,
            )
            _end_phase("evidence_eval")
            
            # Propagate verdicts to deferred claims (evidence sharing)
            if target_selection.evidence_sharing and deferred_claims:
                result = propagate_deferred_verdicts(
                    result=result,
                    evidence_sharing=target_selection.evidence_sharing,
                    deferred_claims=deferred_claims,
                    calibration_registry=self._calibration_registry,
                )
            
            locale_decisions = getattr(search_state, "locale_decisions", {}) or {}
            _start_phase("verdict")
            locale_payload = None
            if anchor_claim_id and anchor_claim_id in locale_decisions:
                locale_payload = locale_decisions[anchor_claim_id]
            elif len(locale_decisions) == 1:
                locale_payload = next(iter(locale_decisions.values()))
            if locale_payload:
                result["locale_decision"] = locale_payload
                audit = result.get("audit") or {}
                audit["locale_decision"] = locale_payload
                result["audit"] = audit

            # Construct RGBA for frontend
            # [danger, verified, context/explainability, style]
            r_score = float(result.get("danger_score", -1.0))
            g_score = float(result.get("verified_score", -1.0))
            b_score = float(result.get("explainability_score", -1.0))
            if b_score < 0:
                b_score = float(result.get("context_score", -1.0))
            a_score = float(result.get("style_score", -1.0))
            
            # Normalize -1.0 to 0.0 for RGBA (visuals) or keep -1.0 if frontend handles it?
            # Standard RGBA usually expects 0.0-1.0. 
            # If missing, we use 0.0 to avoid glitches, or 0.5 for neutral.
            # Let's use max(0.0, score).
            result["rgba"] = [
                max(0.0, r_score),
                max(0.0, g_score),
                max(0.0, b_score),
                max(0.0, a_score),
            ]
            
            Trace.event("pipeline.run.completed", {"outcome": "scored"})
            _end_phase("verdict")
            return _attach_cost_summary(result)
        finally:
            self.agent.llm_client._meter = prior_llm_meter
            self.search_mgr.web_tool._tavily._meter = prior_tavily_meter

    def estimate_cost_range(
        self,
        *,
        claim_count: int,
        search_count: int,
        search_type: str,
    ) -> dict:
        policy = load_pricing_policy()
        estimator = CostEstimator(
            policy,
            standard_model="gpt-5-nano-2025-08-07",
            pro_model="gpt-5",
            standard_search_credits=1,
            pro_search_credits=2,
        )
        # Treat advanced/pro searches as higher cost estimates.
        if search_type in ("advanced", "pro"):
            return estimator.estimate_range(
                claim_count=claim_count,
                search_count=max(1, search_count),
            )
        return estimator.estimate_range(
            claim_count=claim_count,
            search_count=max(1, search_count),
        )

    async def _prepare_input(
        self,
        *,
        fact: str,
        preloaded_context: str | None,
        preloaded_sources: list | None,
        needs_cleaning: bool,
        source_url: str | None,
        progress_callback,
    ) -> _PreparedInput:
        final_context = preloaded_context or ""
        final_sources = preloaded_sources or []

        original_fact = fact
        inline_sources: list[dict] = []
        exclude_url = source_url

        if self._is_url(fact) and not preloaded_context:
            exclude_url = fact
            fetched_text = await self._resolve_url_content(fact)
            if fetched_text:
                url_anchors = self._extract_url_anchors(fetched_text, exclude_url=exclude_url)
                if url_anchors:
                    logger.debug("[Pipeline] Found %d URL-anchor pairs in raw text", len(url_anchors))

                if len(fetched_text) > 10000 and progress_callback:
                    await progress_callback("processing_large_text")
                    logger.debug(
                        "[Pipeline] Large text detected: %d chars, extended timeout",
                        len(fetched_text),
                    )

                budgeted_fetched, _ = self._apply_content_budget(fetched_text, source="url_fetched")
                cleaned_article = await self.agent.clean_article(budgeted_fetched)
                fact = cleaned_article or budgeted_fetched
                final_context = fact

                if url_anchors and cleaned_article:
                    inline_sources = self._restore_urls_from_anchors(cleaned_article, url_anchors)
                    if inline_sources:
                        logger.debug(
                            "[Pipeline] Restored %d inline source candidates after cleaning",
                            len(inline_sources),
                        )
                        Trace.event(
                            "pipeline.inline_sources",
                            {
                                "count": len(inline_sources),
                                "urls": [s["url"][:80] for s in inline_sources[:5]],
                            },
                        )
                        for src in inline_sources:
                            src["is_primary_candidate"] = True

        elif needs_cleaning and not self._is_url(fact):
            logger.debug("[Pipeline] Extension page mode: cleaning %d chars", len(fact))

            url_anchors = self._extract_url_anchors(fact, exclude_url=exclude_url)
            if url_anchors:
                logger.debug("[Pipeline] Found %d URL-anchor pairs in extension text", len(url_anchors))

            if len(fact) > 10000 and progress_callback:
                await progress_callback("processing_large_text")
                logger.debug("[Pipeline] Large text detected: %d chars, extended timeout", len(fact))

            budgeted_fact, _ = self._apply_content_budget(fact, source="extension")
            cleaned_article = await self.agent.clean_article(budgeted_fact)
            if cleaned_article:
                if url_anchors:
                    inline_sources = self._restore_urls_from_anchors(cleaned_article, url_anchors)
                    if inline_sources:
                        logger.debug(
                            "[Pipeline] Restored %d inline source candidates after cleaning",
                            len(inline_sources),
                        )
                        Trace.event(
                            "pipeline.inline_sources",
                            {
                                "count": len(inline_sources),
                                "urls": [s["url"][:80] for s in inline_sources[:5]],
                            },
                        )
                        for src in inline_sources:
                            src["is_primary_candidate"] = True

                fact = cleaned_article
                final_context = fact
            else:
                fact = budgeted_fact
                final_context = final_context or fact

        elif not self._is_url(fact) and not needs_cleaning:
            url_anchors = self._extract_url_anchors(fact, exclude_url=exclude_url)
            if url_anchors:
                logger.debug("[Pipeline] Found %d URL-anchor pairs in plain text", len(url_anchors))
                for item in url_anchors:
                    inline_sources.append(
                        {
                            "url": item["url"],
                            "title": item["anchor"],
                            "domain": item["domain"],
                            "source_type": "inline",
                            "is_trusted": False,
                            "is_primary_candidate": True,
                        }
                    )
                if inline_sources:
                    Trace.event(
                        "pipeline.inline_sources",
                        {
                            "count": len(inline_sources),
                            "urls": [s["url"][:80] for s in inline_sources[:5]],
                        },
                    )

        return _PreparedInput(
            fact=fact,
            original_fact=original_fact,
            final_context=final_context,
            final_sources=final_sources,
            inline_sources=inline_sources,
        )

    async def _extract_claims(
        self,
        *,
        fact: str,
        lang: str,
        progress_callback,
    ) -> tuple[list, bool, str, str, str]:
        if progress_callback:
            await progress_callback("extracting_claims")

        fact_first_line = fact.strip().split("\n")[0]
        blob = fact_first_line if len(fact_first_line) > 20 else fact[:200]

        cleaned_fact = clean_article_text(fact)
        task_claims = asyncio.create_task(self.agent.extract_claims(cleaned_fact, lang=lang))

        if len(blob) > 150:
            blob = blob[:150].rsplit(" ", 1)[0]
        fast_query = normalize_search_query(blob)

        if progress_callback:
            await progress_callback("extracting_claims")

        try:
            claims_result = await task_claims
            if isinstance(claims_result, tuple):
                if len(claims_result) >= 4:
                    claims, should_check_oracle, article_intent, stitched_text = claims_result[:4]
                elif len(claims_result) == 3:
                    claims, should_check_oracle, article_intent = claims_result
                    stitched_text = ""
                elif len(claims_result) == 2:
                    claims, should_check_oracle = claims_result
                    article_intent = "news"
                    stitched_text = ""
                else:
                    claims, should_check_oracle, article_intent, stitched_text = claims_result, False, "news", ""
            else:
                claims, should_check_oracle, article_intent, stitched_text = [], False, "news", ""
        except asyncio.CancelledError:
            claims = []
            should_check_oracle = False
            article_intent = "news"
            stitched_text = ""

        if claims:
            for i, c in enumerate(claims):
                c["id"] = f"c{i+1}"

        return claims, should_check_oracle, article_intent, fast_query, stitched_text

    async def _verify_inline_sources(
        self,
        *,
        inline_sources: list[dict],
        claims: list,
        fact: str,
        final_sources: list,
        progress_callback,
    ) -> list:
        if inline_sources and claims:
            if progress_callback:
                await progress_callback("verifying_sources")

            article_excerpt = fact[:500] if fact else ""
            verified_inline_sources = []

            verification_tasks = [
                self.agent.verify_inline_source_relevance(claims, src, article_excerpt)
                for src in inline_sources
            ]
            verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)

            for src, result in zip(inline_sources, verification_results):
                if isinstance(result, Exception):
                    logger.warning("[Pipeline] Inline source verification failed: %s", result)
                    src["is_primary"] = False
                    src["is_relevant"] = True
                    verified_inline_sources.append(src)
                    continue

                is_relevant = result.get("is_relevant", True)
                is_primary = result.get("is_primary", False)

                if not is_relevant:
                    logger.debug("[Pipeline] Inline source rejected: %s", src.get("domain"))
                    continue

                src["is_primary"] = is_primary
                src["is_relevant"] = True
                if is_primary:
                    src["is_trusted"] = True
                verified_inline_sources.append(src)
                logger.debug(
                    "[Pipeline] Inline source %s: relevant=%s, primary=%s",
                    src.get("domain"),
                    is_relevant,
                    is_primary,
                )

            if verified_inline_sources:
                if hasattr(self.search_mgr, "apply_evidence_acquisition_ladder"):
                    verified_inline_sources = await self.search_mgr.apply_evidence_acquisition_ladder(
                        verified_inline_sources
                    )
                Trace.event(
                    "pipeline.inline_sources_verified",
                    {
                        "total": len(inline_sources),
                        "passed": len(verified_inline_sources),
                        "primary": len([s for s in verified_inline_sources if s.get("is_primary")]),
                    },
                )
                
                # Fetch content for primary inline sources
                primary_sources = [s for s in verified_inline_sources if s.get("is_primary")]
                if primary_sources and self.search_mgr:
                    for src in primary_sources[:2]:  # Limit to 2 primary sources
                        url = src.get("url", "")
                        if url and not src.get("content"):
                            try:
                                fetched = await self.search_mgr.web_tool.fetch_page_content(url)
                                if fetched and len(fetched) > 100:
                                    budgeted_content, _ = self._apply_content_budget(
                                        fetched, source="inline_source"
                                    )
                                    src["content"] = budgeted_content
                                    src["snippet"] = budgeted_content[:300]
                                    Trace.event("inline_source.content_fetched", {
                                        "url": url[:80],
                                        "chars": len(fetched),
                                    })
                            except Exception as e:
                                logger.debug("[Pipeline] Failed to fetch inline source: %s", e)
                
                final_sources.extend(verified_inline_sources)
            return final_sources

        if inline_sources:
            logger.debug(
                "[Pipeline] No claims extracted, adding %d inline sources as secondary",
                len(inline_sources),
            )
            for src in inline_sources:
                src["is_primary"] = False
                src["is_relevant"] = True
            final_sources.extend(inline_sources)

        return final_sources

    def _extract_url_anchors(self, text: str, exclude_url: str | None = None) -> list[dict]:
        """Extract URL-anchor pairs from article text.
        
        Finds URLs with their surrounding context (anchor text) that can be
        used to identify if the URL reference survives LLM cleaning.
        
        Args:
            text: Raw article text with URLs
            exclude_url: URL to exclude (e.g., the article's own URL)
            
        Returns:
            List of dicts with 'url', 'anchor', and 'domain' keys
        """
        return extract_url_anchors(text, exclude_url=exclude_url)
    
    def _restore_urls_from_anchors(self, cleaned_text: str, url_anchors: list[dict]) -> list[dict]:
        """Find which URL anchors survived cleaning and return them as sources.
        
        Args:
            cleaned_text: LLM-cleaned article text
            url_anchors: List of URL-anchor pairs from _extract_url_anchors
            
        Returns:
            List of source dicts for anchors that survived in cleaned text
        """
        return restore_urls_from_anchors(cleaned_text, url_anchors)


    def _is_url(self, text: str) -> bool:
        return is_url_input(text)

    async def _resolve_url_content(self, url: str) -> str | None:
        """Fetch URL content via Tavily Extract. Cleaning happens in claim extraction."""
        return await resolve_url_content(self.search_mgr, url, log=logger)

    # ─────────────────────────────────────────────────────────────────────────
    # Topic-Aware Round-Robin Query Selection ("Coverage Engine")
    # ─────────────────────────────────────────────────────────────────────────
    
    def _select_diverse_queries(
        self,
        claims: list,
        max_queries: int = 3,
        fact_fallback: str = ""
    ) -> list[str]:
        max_queries = resolve_budgeted_max_queries(claims, default_max=max_queries)
        if max_queries <= 0:
            return []
        return select_diverse_queries(
            claims,
            max_queries=max_queries,
            fact_fallback=fact_fallback,
            log=logger,
        )
    
    def _get_query_by_role(self, claim: dict, role: str) -> str | None:
        """
        Extract query with specific role from claim's query_candidates.
        
        Args:
            claim: Claim dict with query_candidates and/or search_queries
            role: Query role ("CORE", "NUMERIC", "ATTRIBUTION", "LOCAL")
            
        Returns:
            Query text or None if not found
        """
        return get_query_by_role(claim, role)
    
    def _normalize_and_sanitize(self, query: str) -> str | None:
        """
        Normalize query.
        
        Note: Strict gambling keywords removal is deprecated (M64).
        We rely on LLM constraints and Tavily 'topic="news"' mode 
        to prevent gambling/spam results instead of hardcoded stoplists.
        
        Args:
            query: Raw query text
            
        Returns:
            Normalized query or None if invalid
        """
        return normalize_and_sanitize(query)
    
    def _is_fuzzy_duplicate(self, query: str, existing: list[str], threshold: float = 0.9) -> bool:
        """
        Check if query is >threshold similar to any existing query.
        
        Uses Jaccard similarity on word sets.
        
        Args:
            query: Query to check
            existing: List of already-selected queries
            threshold: Similarity threshold (0.9 = 90% word overlap)
            
        Returns:
            True if query is a duplicate
        """
        return is_fuzzy_duplicate(query, existing, threshold=threshold, log=logger)


    def _can_add_search(self, model, search_type, max_cost):
        # Speculative cost check
        current = self.search_mgr.calculate_cost(model, search_type)
        # Add cost of 1 search
        step_cost = int(SEARCH_COSTS.get(search_type, 80))
        return self.search_mgr.can_afford(current + step_cost, max_cost)

    def _finalize_oracle(self, oracle_res: dict, fact: str) -> dict:
        """Format oracle result for return (legacy)."""
        oracle_res["text"] = fact
        oracle_res["search_meta"] = self.search_mgr.get_search_meta()
        return oracle_res

    async def _finalize_oracle_hybrid(
        self, 
        oracle_result: dict, 
        fact: str, 
        lang: str = "en", 
        progress_callback=None
    ) -> dict:
        """
        Format Oracle JACKPOT result for immediate return.
        Added lang parameter for localization support.
        Added granular progress updates (localizing_content).
        
        Converts OracleCheckResult to FactCheckResponse format.
        """
        oracle_result.get("status", "MIXED")
        rating = oracle_result.get("rating", "")
        publisher = oracle_result.get("publisher", "Fact Check")
        url = oracle_result.get("url", "")
        claim_reviewed = oracle_result.get("claim_reviewed", "")
        summary = oracle_result.get("summary", "")
        
        # Use LLM-determined scores from OracleValidationSkill (no heuristics!)
        verified_score = float(oracle_result.get("verified_score", -1.0))
        danger_score = float(oracle_result.get("danger_score", -1.0))
        
        # Fallback for tests / legacy Oracle results without LLM-derived scores.
        if verified_score < 0:
            status_norm = str(oracle_result.get("status", "") or "").upper()
            rating_norm = str(oracle_result.get("rating", "") or "").upper()
            marker = status_norm or rating_norm
            if any(x in marker for x in ("REFUTED", "FALSE", "INCORRECT", "PANTS_ON_FIRE")):
                verified_score = 0.1
            elif any(x in marker for x in ("TRUE", "SUPPORTED", "CORRECT")):
                verified_score = 0.9
            else:
                verified_score = 0.5
        if danger_score < 0:
            danger_score = 0.0
        
        # Build response (English first)
        analysis = f"According to {publisher}, this claim is rated as '{rating}'. {summary}"
        rationale = f"Fact check by {publisher}: Rated as '{rating}'. {claim_reviewed}"
        
        # Translate if non-English and translation_service available
        if lang and lang.lower() not in ("en", "en-us") and self.translation_service:
            if progress_callback:
                await progress_callback("localizing_content")
            
            try:
                analysis = await self.translation_service.translate(analysis, target_lang=lang)
                rationale = await self.translation_service.translate(rationale, target_lang=lang)
            except Exception as e:
                logger.warning("[Pipeline] Translation failed for Oracle result: %s", e)
                # Keep English if translation fails
        
        sources = [{
            "title": f"Fact Check by {publisher}",
            "link": url,
            "url": url,
            "domain": get_registrable_domain(url) if url else publisher.lower().replace(" ", ""),
            "snippet": f"Rating: {rating}. {summary}",
            "origin": "GOOGLE_FACT_CHECK",
            "source_type": "fact_check",
            "is_trusted": True,
        }]
        
        return {
            "verified_score": verified_score,
            "confidence_score": 1.0,
            "danger_score": danger_score,
            "context_score": 1.0,
            "style_score": 1.0,
            "analysis": analysis,
            "rationale": rationale,
            "sources": sources,
            "cost": 0,  # Oracle is free!
            "rgba": [danger_score, verified_score, 1.0, 1.0],
            "text": fact,
            "search_meta": self.search_mgr.get_search_meta(),
            "oracle_jackpot": True,  # Flag for frontend
            "details": [],
        }

    def _create_oracle_source(self, oracle_result: dict) -> dict:
        """
        Create source dict from Oracle result for EVIDENCE scenario.
        
        This source is added to the evidence pack as a Tier A (high trust) source.
        """
        url = oracle_result.get("url", "")
        publisher = oracle_result.get("publisher", "Fact Check")
        rating = oracle_result.get("rating", "")
        claim_reviewed = oracle_result.get("claim_reviewed", "")
        summary = oracle_result.get("summary", "")
        relevance = oracle_result.get("relevance_score", 0.0)
        status = oracle_result.get("status", "MIXED")
        
        return {
            "url": url,
            "domain": get_registrable_domain(url) if url else publisher.lower().replace(" ", ""),
            "title": f"Fact Check: {claim_reviewed[:50]}..." if len(claim_reviewed) > 50 else f"Fact Check: {claim_reviewed}",
            "content": f"{publisher} rated this claim as '{rating}': {summary}",
            "snippet": f"Rating: {rating}. {summary[:200]}",
            "source_type": "fact_check",
            "is_trusted": True,
            "origin": "GOOGLE_FACT_CHECK",
            # Oracle metadata for transparency in scoring
            "oracle_metadata": {
                "relevance_score": relevance,
                "status": status,
                "publisher": publisher,
                "rating": rating,
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Schema-First Query Generation (Assertion-Based)
    # ─────────────────────────────────────────────────────────────────────────

    def _select_queries_from_claim_units(
        self,
        claim_units: list,
        max_queries: int = 3,
        fact_fallback: str = "",
    ) -> list[str]:
        """
        Generate search queries from structured ClaimUnits.
        
        Key difference from legacy:
        - Only FACT assertions generate verification queries
        - CONTEXT assertions are informational (no refutation search)
        - Queries are built from assertion_key + value
        
        Args:
            claim_units: List of ClaimUnit objects (from schema)
            max_queries: Maximum queries to return
            fact_fallback: Fallback text if no queries generated
            
        Returns:
            List of search queries
        """
        return select_queries_from_claim_units(
            claim_units,
            max_queries=max_queries,
            fact_fallback=fact_fallback,
            log=logger,
        )

    def _build_assertion_query(self, unit, assertion) -> str | None:
        """
        Build search query for a specific assertion.
        
        Query structure: "{subject} {assertion.value} {context}"
        
        Examples:
        - event.location.city: "Joshua Paul fight Miami official location"
        - numeric.value: "Bitcoin price $42000 official"
        - event.time: "Joshua Paul fight March 2025 date confirmed"
        
        Args:
            unit: ClaimUnit containing the assertion
            assertion: Assertion to build query for
            
        Returns:
            Search query string or None
        """
        return build_assertion_query(unit, assertion)

    def _get_claim_units_for_evidence_mapping(
        self,
        claim_units: list,
        sources: list[dict],
    ) -> dict[str, list[str]]:
        """
        Map sources to assertion_keys for targeted verification.
        
        This is used by clustering to understand which assertion
        each piece of evidence relates to.
        
        Returns:
            Dict of claim_id -> list of assertion_keys that need evidence
        """
        return get_claim_units_for_evidence_mapping(claim_units, sources)
