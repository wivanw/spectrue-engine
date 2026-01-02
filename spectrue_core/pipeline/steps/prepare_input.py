# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from spectrue_core.pipeline.core import PipelineContext
from spectrue_core.pipeline.errors import PipelineExecutionError
from spectrue_core.utils.trace import Trace

logger = logging.getLogger(__name__)


@dataclass
class PrepareInputStep:
    """
    Prepare and clean input text for processing.

    Handles:
    - URL content extraction (if input is URL)
    - Text cleaning/normalization
    - LLM-based cleaning for extension content
    - Inline source separation

    Context Input:
        - claims[0].text or first claim text as fact
        - extras: preloaded_context, preloaded_sources, source_url, needs_cleaning

    Context Output:
        - extras: prepared_fact, prepared_context, inline_sources
    """

    agent: Any  # FactCheckerAgent
    search_mgr: Any  # SearchManager
    config: Any  # SpectrueConfig
    name: str = "prepare_input"

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Prepare input text."""
        from spectrue_core.verification.pipeline.pipeline_input import (
            is_url_input,
            resolve_url_content,
            extract_url_anchors,
            restore_urls_from_anchors,
            apply_content_budget,
        )
        from spectrue_core.runtime_config import ContentBudgetConfig

        try:
            # Get fact from claims or context
            fact = ctx.get_extra("raw_fact", "")
            if not fact and ctx.claims:
                fact = ctx.claims[0].get("text", "")

            preloaded_context = ctx.get_extra("preloaded_context")
            preloaded_sources = ctx.get_extra("preloaded_sources")
            source_url = ctx.get_extra("source_url")
            needs_cleaning = ctx.get_extra("needs_cleaning", False)
            progress_callback = ctx.get_extra("progress_callback")

            # Extract config
            content_budget_cfg = getattr(getattr(self.config, "runtime", None), "content_budget", None)
            if not isinstance(content_budget_cfg, ContentBudgetConfig):
                content_budget_cfg = ContentBudgetConfig()

            # Logic ported from ValidationPipeline._prepare_input
            final_context = preloaded_context or ""
            final_sources = list(preloaded_sources) if preloaded_sources else []
            inline_sources_list: list[dict] = []
            exclude_url = source_url

            is_url = is_url_input(fact)

            if is_url and not preloaded_context:
                exclude_url = fact
                fetched_text = await resolve_url_content(self.search_mgr, fact)
                
                if fetched_text:
                    url_anchors = extract_url_anchors(fetched_text, exclude_url=exclude_url)
                    if url_anchors:
                        logger.debug("[PrepareInputStep] Found %d URL-anchor pairs in raw text", len(url_anchors))

                    if len(fetched_text) > 10000 and progress_callback:
                        await progress_callback("processing_large_text")

                    budgeted_fetched, _ = apply_content_budget(
                        fetched_text, content_budget_cfg, source="url_fetched"
                    )
                    cleaned_article = await self.agent.clean_article(budgeted_fetched)
                    fact = cleaned_article or budgeted_fetched
                    final_context = fact

                    if url_anchors and cleaned_article:
                        inline_sources_list = restore_urls_from_anchors(cleaned_article, url_anchors)
                        for src in inline_sources_list:
                            src["is_primary_candidate"] = True

            elif needs_cleaning and not is_url:
                logger.debug("[PrepareInputStep] Extension page mode: cleaning %d chars", len(fact))
                url_anchors = extract_url_anchors(fact, exclude_url=exclude_url)
                
                if len(fact) > 10000 and progress_callback:
                    await progress_callback("processing_large_text")

                budgeted_fact, _ = apply_content_budget(fact, content_budget_cfg, source="extension")
                cleaned_article = await self.agent.clean_article(budgeted_fact)
                
                if cleaned_article:
                    if url_anchors:
                        inline_sources_list = restore_urls_from_anchors(cleaned_article, url_anchors)
                        for src in inline_sources_list:
                            src["is_primary_candidate"] = True
                    fact = cleaned_article
                    final_context = fact
                else:
                    fact = budgeted_fact
                    final_context = final_context or fact

            elif not is_url and not needs_cleaning:
                url_anchors = extract_url_anchors(fact, exclude_url=exclude_url)
                if url_anchors:
                    logger.debug("[PrepareInputStep] Found %d URL-anchor pairs in plain text", len(url_anchors))
                    for item in url_anchors:
                        inline_sources_list.append({
                            "url": item["url"],
                            "title": item["anchor"],
                            "domain": item["domain"],
                            "source_type": "inline",
                            "is_trusted": False,
                            "is_primary_candidate": True,
                        })

            Trace.event(
                "prepare_input.completed",
                {
                    "fact_len": len(fact),
                    "has_context": bool(final_context),
                    "sources_count": len(final_sources),
                    "inline_count": len(inline_sources_list),
                },
            )

            return (
                ctx.set_extra("prepared_fact", fact)
                .set_extra("original_fact", ctx.get_extra("raw_fact", "") or fact) # Fallback to new fact as original if missing
                .set_extra("prepared_context", final_context)
                .set_extra("prepared_sources", final_sources)
                .set_extra("inline_sources", inline_sources_list)
            )

        except Exception as e:
            logger.exception("[PrepareInputStep] Failed: %s", e)
            raise PipelineExecutionError(self.name, str(e), cause=e) from e
