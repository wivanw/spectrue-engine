# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations
import logging
from typing import Any

from spectrue_core.verification.search.search_mgr import SearchManager
from spectrue_core.verification.calibration.calibration_registry import CalibrationRegistry
from spectrue_core.utils.embedding_service import EmbedService
from spectrue_core.config import SpectrueConfig
from spectrue_core.runtime_config import ContentBudgetConfig
from spectrue_core.agents.fact_checker_agent import FactCheckerAgent
from spectrue_core.graph import ClaimGraphBuilder
from spectrue_core.verification.pipeline.pipeline_input import apply_content_budget
from spectrue_core.pipeline.factory import PipelineFactory
from spectrue_core.pipeline.dag import PipelineContext
import time

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Orchestrates the fact-checking waterfall process.
    """
    def __init__(self, config: SpectrueConfig, agent: FactCheckerAgent, translation_service=None, search_mgr=None):
        self.config = config
        self.agent = agent
        self._calibration_registry = CalibrationRegistry.from_runtime(
            getattr(config, "runtime", None)
        )
        EmbedService.configure(openai_api_key=getattr(config, "openai_api_key", None))
        # Pass oracle_skill to SearchManager for hybrid mode (or use injected mgr)
        self.search_mgr = search_mgr or SearchManager(config, oracle_validator=agent.oracle_skill)
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
        return apply_content_budget(text, self._content_budget_config, source=source)

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
        *,
        search_type: str = "smart",
        gpt_model: str = "standard",
        preloaded_sources: list | None = None,
        preloaded_context: str | None = None,
        lang: str = "en",
        source_url: str | None = None,
        needs_cleaning: bool = False,
        claim_count: int | None = None,
        search_count: int | None = None,
        runtime_config: Any | None = None,
        progress_callback: Any | None = None,
        preloaded_claims: list | None = None,
        extract_claims_only: bool = False,
    ) -> dict:
        """
        Execute the validation pipeline using DAG architecture.
        """
        from spectrue_core.pipeline.mode import get_mode
        
        start_time = time.time()
        
        # Capture prior meters for restoration
        prior_llm_meter = getattr(self.agent.llm_client, "_meter", None)
        prior_tavily_meter = getattr(self.search_mgr.web_tool._tavily, "_meter", None)
        
        # Setup progress
        if progress_callback:
            try:
                await progress_callback("analyzing_input")
            except Exception:
                pass

        # Determine Mode
        mode_name = "normal"
        if runtime_config:
            if isinstance(runtime_config, dict):
                 mode_name = runtime_config.get("profile", "normal")
            elif hasattr(runtime_config, "profile"):
                 mode_name = runtime_config.profile
                 
        try:
            # Initialize Context
            mode = get_mode(mode_name)
            ctx = PipelineContext(
                mode=mode,
                lang=lang,
                search_type=search_type,
                gpt_model=gpt_model,
            )
            
            # Populate Context
            # Setup extras
            ctx = (ctx
                .set_extra("start_time", start_time)
                .set_extra("raw_fact", fact)
                .set_extra("preloaded_context", preloaded_context)
                .set_extra("preloaded_sources", preloaded_sources)
                .set_extra("source_url", source_url)
                .set_extra("needs_cleaning", needs_cleaning)
                .set_extra("claim_count_hint", claim_count)
                .set_extra("search_count_hint", search_count)
                .set_extra("progress_callback", progress_callback)
                .set_extra("runtime_config", runtime_config)
                .set_extra("preloaded_claims", preloaded_claims)
                .set_extra("extract_claims_only", extract_claims_only)
            )

            # Build DAG (with extraction logic if needed)
            dag = PipelineFactory(search_mgr=self.search_mgr, agent=self.agent).build(
                mode_name, config=self.config, extraction_only=extract_claims_only
            )
            
            # Execute DAG
            result_ctx = await dag.run(ctx)
            
            # Assemble Result
            # ResultAssemblyStep puts final payload in "final_result"
            if result_ctx.get_extra("final_result"):
                return result_ctx.get_extra("final_result")

            # Fallback (should not happen if ResultAssemblyStep runs)
            payload = result_ctx.verdict or {}
            metering = result_ctx.get_extra("metering")
            
            return attach_cost_summary(payload, metering=metering)

        except Exception as e:
            logger.exception("[ValidationPipeline] DAG execution failed: %s", e)
            
            error_payload = {
                "status": "error",
                "error": str(e),
                "verdict": "error",
                "verified_score": 0.0,
                "rgba": None
            }
            
            try:
                if 'ctx' in locals() and ctx:
                    metering = ctx.get_extra("metering")
                    if metering:
                            metering.phase_tracker.record_reason("execution_error", str(e))
                            return attach_cost_summary(error_payload, metering=metering)
            except:
                pass
                
            return error_payload

        finally:
            # Restore meters
            if prior_llm_meter:
                self.agent.llm_client._meter = prior_llm_meter
            if prior_tavily_meter:
                if getattr(self.search_mgr, "web_tool", None) and getattr(self.search_mgr.web_tool, "_tavily", None):
                    self.search_mgr.web_tool._tavily._meter = prior_tavily_meter


