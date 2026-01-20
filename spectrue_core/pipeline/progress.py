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
"""
Progress Estimation Logic

Calculates weighted progress for DAG execution and emits rich events.
"""

from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, Optional
import logging



@dataclass
class ProgressEvent:
    """
    Rich progress event emitted during pipeline execution.
    """
    percent: int
    status_key: str  # Localization key for the status message
    status_detail_key: Optional[str] = None  # Localization key for detailed description
    meta: Dict[str, str] = field(default_factory=dict)  # Dynamic values for the message (e.g. {processed: 10, total: 20})


# Step weights define how much "progress" each step represents out of 100%
# Total should ideally sum to 100, but logic will clamp/normalize if needed.
STEP_WEIGHTS = {
    # Setup & Extraction (15%)
    "metering_setup": 1,
    "extract_claims": 14,

    # Search & Retrieval (40%)
    "build_queries": 5,
    "web_search": 25,  # Matches WebSearchStep.name
    "verify_inline_sources": 5,
    "fetch_chunks": 5,

    # Evidence Processing (20%)
    "evidence_collect": 10,  # Matches EvidenceCollectStep
    "cluster_evidence": 10,  # Matches ClusterEvidenceStep

    # Reasoning & Scoring (20%)
    "summarize_evidence": 5,
    "judge_claims": 15,
    "judge_standard": 15,  # Mutually exclusive with judge_claims

    # Finalization (5%)
    "assemble_standard_result": 5,
    "assemble_deep_result": 5,
}

# Mapping step names to user-facing status keys
STATUS_KEYS = {
    "metering_setup": "status.initializing",
    "extract_claims": "loader.extracting_claims",
    "claim_graph": "loader.building_claim_graph",
    "oracle_flow": "loader.checking_oracle",
    "build_queries": "loader.generating_queries",
    "web_search": "loader.searching_phase_a",  # Generic initial search
    "verify_inline_sources": "loader.verifying_sources",
    "fetch_chunks": "loader.processing_sources",
    "cluster_evidence": "loader.clustering_evidence",
    "evidence_collect": "loader.analyzing_evidence",
    "stance_annotate": "loader.stance_annotation",
    "summarize_evidence": "loader.compressing_context",
    "judge_claims": "loader.verifying_claim",  # Deep mode specific
    "judge_standard": "loader.ai_analysis",
    "assemble_standard_result": "loader.finalizing",
    "assemble_deep_result": "loader.finalizing",
}


logger = logging.getLogger(__name__)

class ProgressEstimator:
    """
    Estimates progress percentage based on completed DAG steps.
    """

    def __init__(self, callback: Callable[[ProgressEvent], Awaitable[None]]):
        self.callback = callback
        self.completed_weight = 0.0
        self.total_weight = sum(STEP_WEIGHTS.values())
        self.executed_steps: set[str] = set()

    async def on_step_start(self, step_name: str):
        """Called when a step starts."""
        try:
            logger.debug(f"ProgressEstimator.on_step_start: step={step_name}, completed_weight={self.completed_weight}")
        except Exception:
            pass
        # We don't advance percentage on start, but we update status text
        current_percent = int((self.completed_weight / self.total_weight) * 95)
        
        status_key = STATUS_KEYS.get(step_name, "status.processing")
        
        event = ProgressEvent(
            percent=max(5, current_percent), # Minimum 5% to show activity
            status_key=status_key,
            status_detail_key=f"{status_key}.desc",
            meta={"step": step_name}
        )
        await self.callback(event)

    async def on_step_end(self, step_name: str):
        """Called when a step finishes successfully."""
        try:
            logger.debug(f"ProgressEstimator.on_step_end: step={step_name}, before_completed_weight={self.completed_weight}")
        except Exception:
            pass
        if step_name in self.executed_steps:
            return
        
        self.executed_steps.add(step_name)
        weight = STEP_WEIGHTS.get(step_name, 0)
        self.completed_weight += weight
        
        # Calculate new percentage
        # Use 95% scale to match on_step_start and reserve room for final "Done" state
        percent = int((self.completed_weight / self.total_weight) * 95)
        # Ensure a minimum visible progress unless it's essentially 0
        # Ensure a minimum visible progress unless it's essentially 0
        percent = max(5, percent)
        
        try:
            logger.debug(f"ProgressEstimator.on_step_end: step={step_name}, added_weight={weight}, new_completed_weight={self.completed_weight}, percent={percent}")
        except Exception:
            pass
        
        # Emit an explicit progress event on step end so frontend updates percentage
        status_key = STATUS_KEYS.get(step_name, "status.processing")
        event = ProgressEvent(
            percent=percent,
            status_key=status_key,
            status_detail_key=f"{status_key}.desc",
            meta={"step": step_name, "completed": True},
        )
        try:
            await self.callback(event)
        except Exception:
            # Swallow to keep pipeline robust
            pass
        
        # We don't emit an event here usually, the next step start will update the UI
        # unless it's the very last step.
