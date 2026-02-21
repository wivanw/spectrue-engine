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
from typing import Callable, Awaitable, Dict, Optional, Any
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


# Progress emitter uses attributes from Step objects (weight, status_key).
# Refer to spectrue_core.pipeline.core.Step protocol for defaults.


logger = logging.getLogger(__name__)

class ProgressEstimator:
    """
    Estimates progress percentage based on completed DAG steps.
    """

    def __init__(self, callback: Callable[[ProgressEvent], Awaitable[None]], expected_total_weight: float = 0.0):
        self.callback = callback
        self.completed_weight = 0.0
        self.total_weight = expected_total_weight
        self.executed_steps: set[str] = set()
        self.last_status_key: Optional[str] = None
        self.last_percent: int = 0
        self.step_objects: dict[str, Any] = {} # Map name -> Step object
        self.current_step: Optional[str] = None

    def set_planned_nodes(self, nodes: list[Any]):
        """Sets the list of nodes (StepNode) that are expected to run."""
        self.step_objects = {n.name: n.step for n in nodes}
        
        total = 0.0
        for name, step in self.step_objects.items():
            # Try to get weight from step object, default to 1.0
            weight = getattr(step, "weight", 1.0)
            total += weight
            
        # Use max to avoid jumping backwards if we pre-estimated more
        self.total_weight = max(self.total_weight, total)
        logger.info(f"ProgressEstimator: planned steps={len(nodes)}, total_weight={self.total_weight} (calculated={total})")

    def set_planned_steps(self, step_names: list[str]):
        """Legacy compatibility method. Discouraged in DAG mode."""
        self.total_weight = max(self.total_weight, float(len(step_names)))
        logger.warning(f"ProgressEstimator(legacy): planned steps={len(step_names)}, total_weight={self.total_weight}")

    async def on_step_start(self, step_name: str):
        """Called when a step starts."""
        self.current_step = step_name
        try:
            logger.debug(f"ProgressEstimator.on_step_start: step={step_name}, completed_weight={self.completed_weight}")
        except Exception:
            pass
        # Use 100% scale for progress, clamp to 95% until finalized
        divisor = max(1.0, self.total_weight)
        current_percent = int((self.completed_weight / divisor) * 95)
        
        # Automatically generate status key from step name
        status_key = f"loader.{step_name}"
        
        # 'verifying_claims' is a virtual step used in Deep Mode to group multiple DAG steps
        is_virtual = step_name in {"verifying_claims", "extracting_claims", "extract_claims"}
        
        if self.step_objects.get(step_name) is None and not is_virtual:
            logger.warning(f"[Progress] Unknown step_name '{step_name}' - no step object")
        
        # Avoid redundant events if status and percent haven't changed much
        if status_key == self.last_status_key and current_percent == self.last_percent:
             return
        
        self.last_status_key = status_key
        self.last_percent = current_percent

        event = ProgressEvent(
            percent=max(5, current_percent), # Minimum 5% to show activity
            status_key=status_key,
            status_detail_key=f"{status_key}.desc",
            meta={"step": step_name}
        )
        await self.callback(event)

    async def on_step_progress(self, step_name: str, processed: int, total: int, **kwargs):
        """Called for sub-progress within a step (e.g. per-claim processing)."""
        if total <= 0:
            return
            
        # If we got sub-progress for a different step than we thought we were in, update it
        if self.current_step != step_name:
            self.current_step = step_name
            
        step_obj = self.step_objects.get(step_name)
        weight = getattr(step_obj, "weight", 1.0 if step_obj else 0.0)
        
        if weight <= 0:
            return
            
        # Calculate fractional progress within the current step
        fraction = min(1.0, processed / total)
        sub_weight = fraction * weight
        
        divisor = max(1.0, self.total_weight)
        percent = int(((self.completed_weight + sub_weight) / divisor) * 95)
        percent = max(5, percent)
        
        # Only emit if percentage or status changed decently
        status_key = f"loader.{step_name}"
        if status_key == self.last_status_key and percent <= self.last_percent:
            return
            
        self.last_status_key = status_key
        self.last_percent = percent
        
        event = ProgressEvent(
            percent=percent,
            status_key=status_key,
            status_detail_key=f"{status_key}.desc",
            meta={"step": step_name, "processed": processed, "total": total, **kwargs}
        )
        try:
            await self.callback(event)
        except Exception:
            pass

    async def on_step_end(self, step_name: str):
        """Called when a step finishes successfully."""
        try:
            logger.debug(f"ProgressEstimator.on_step_end: step={step_name}, before_completed_weight={self.completed_weight}")
        except Exception:
            pass
        if step_name in self.executed_steps:
            return
        
        self.executed_steps.add(step_name)
        
        # Get weight from step object with fallback
        step_obj = self.step_objects.get(step_name)
        weight = getattr(step_obj, "weight", 1.0 if step_obj else 0.0)
        self.completed_weight += weight
        
        # Calculate new percentage
        divisor = max(1.0, self.total_weight)
        percent = int((self.completed_weight / divisor) * 95)
        # Ensure a minimum visible progress
        percent = max(5, percent)
        
        try:
            logger.debug(f"ProgressEstimator.on_step_end: step={step_name}, added_weight={weight}, new_completed_weight={self.completed_weight}, percent={percent}")
        except Exception:
            pass
        
        # Automatically generate status key from step name
        status_key = f"loader.{step_name}"
        self.last_status_key = status_key
        self.last_percent = percent

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
