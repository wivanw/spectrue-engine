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

    def __init__(self, callback: Callable[[ProgressEvent], Awaitable[None]]):
        self.callback = callback
        self.completed_weight = 0.0
        self.total_weight = 0.0
        self.executed_steps: set[str] = set()
        self.last_status_key: Optional[str] = None
        self.step_objects: dict[str, Any] = {} # Map name -> Step object

    def set_planned_nodes(self, nodes: list[Any]):
        """Sets the list of nodes (StepNode) that are expected to run."""
        self.step_objects = {n.name: n.step for n in nodes}
        
        total = 0.0
        for name, step in self.step_objects.items():
            # Try to get weight from step object, default to 1.0
            weight = getattr(step, "weight", 1.0)
            total += weight
            
        self.total_weight = total
        logger.info(f"ProgressEstimator: planned steps={len(nodes)}, total_weight={self.total_weight}")

    def set_planned_steps(self, step_names: list[str]):
        """Legacy compatibility method. Discouraged in DAG mode."""
        self.total_weight = float(len(step_names))
        logger.warning(f"ProgressEstimator(legacy): planned steps={len(step_names)}, total_weight={self.total_weight}")

    async def on_step_start(self, step_name: str):
        """Called when a step starts."""
        try:
            logger.debug(f"ProgressEstimator.on_step_start: step={step_name}, completed_weight={self.completed_weight}")
        except Exception:
            pass
        # Use 100% scale for progress, clamp to 95% until finalized
        divisor = max(1.0, self.total_weight)
        current_percent = int((self.completed_weight / divisor) * 95)
        
        # Automatically generate status key from step name
        status_key = f"loader.{step_name}"
        
        if self.step_objects.get(step_name) is None:
            logger.warning(f"[Progress] Unknown step_name '{step_name}' - no step object")
        
        # Avoid redundant events if status and percent haven't changed much
        if status_key == self.last_status_key:
             return
        
        self.last_status_key = status_key

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
