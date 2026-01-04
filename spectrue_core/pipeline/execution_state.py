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
"""Step-level execution state for DAG pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectrue_core.pipeline.constants import (
    DAG_STEP_STATUS_FAILED,
    DAG_STEP_STATUS_PENDING,
    DAG_STEP_STATUS_RUNNING,
    DAG_STEP_STATUS_SKIPPED,
    DAG_STEP_STATUS_SUCCEEDED,
)


@dataclass
class StepExecutionState:
    """Execution status and timing for a single DAG step."""

    name: str
    status: str = DAG_STEP_STATUS_PENDING
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    error_type: str | None = None
    optional: bool = False
    depends_on: list[str] = field(default_factory=list)
    layer: int | None = None
    skip_reason: str | None = None

    def mark_running(self, *, timestamp: float) -> None:
        self.status = DAG_STEP_STATUS_RUNNING
        self.started_at = timestamp

    def mark_succeeded(self, *, timestamp: float) -> None:
        self.status = DAG_STEP_STATUS_SUCCEEDED
        self.completed_at = timestamp

    def mark_failed(self, *, timestamp: float, error: Exception | str) -> None:
        self.status = DAG_STEP_STATUS_FAILED
        self.completed_at = timestamp
        self.error = str(error)
        self.error_type = error.__class__.__name__ if isinstance(error, Exception) else None

    def mark_skipped(self, *, timestamp: float, reason: str | None = None) -> None:
        self.status = DAG_STEP_STATUS_SKIPPED
        self.completed_at = timestamp
        self.skip_reason = reason

    def to_dict(self) -> dict[str, Any]:
        duration = None
        if self.started_at is not None and self.completed_at is not None:
            duration = self.completed_at - self.started_at
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": duration,
            "error": self.error,
            "error_type": self.error_type,
            "optional": self.optional,
            "depends_on": list(self.depends_on),
            "layer": self.layer,
            "skip_reason": self.skip_reason,
        }


@dataclass
class LayerExecutionState:
    """Timing metadata for a DAG execution layer."""

    index: int
    steps: list[str] = field(default_factory=list)
    started_at: float | None = None
    completed_at: float | None = None

    def mark_started(self, *, timestamp: float) -> None:
        self.started_at = timestamp

    def mark_completed(self, *, timestamp: float) -> None:
        self.completed_at = timestamp

    def to_dict(self) -> dict[str, Any]:
        duration = None
        if self.started_at is not None and self.completed_at is not None:
            duration = self.completed_at - self.started_at
        return {
            "index": self.index,
            "steps": list(self.steps),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": duration,
        }


@dataclass
class DAGExecutionState:
    """Execution state and timing for a DAG pipeline run."""

    steps: dict[str, StepExecutionState] = field(default_factory=dict)
    layers: list[LayerExecutionState] = field(default_factory=list)
    ordered_steps: list[str] = field(default_factory=list)
    started_at: float | None = None
    completed_at: float | None = None

    def ensure_step(
        self,
        name: str,
        *,
        depends_on: list[str],
        optional: bool,
        layer: int | None,
    ) -> StepExecutionState:
        if name not in self.steps:
            self.steps[name] = StepExecutionState(
                name=name,
                optional=optional,
                depends_on=list(depends_on),
                layer=layer,
            )
        return self.steps[name]

    def to_dict(self) -> dict[str, Any]:
        duration = None
        if self.started_at is not None and self.completed_at is not None:
            duration = self.completed_at - self.started_at
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": duration,
            "ordered_steps": list(self.ordered_steps),
            "layers": [layer.to_dict() for layer in self.layers],
            "steps": {name: state.to_dict() for name, state in self.steps.items()},
        }

    def to_summary(self) -> dict[str, Any]:
        """Return a lightweight summary for logs or UI."""
        duration = None
        if self.started_at is not None and self.completed_at is not None:
            duration = self.completed_at - self.started_at
        return {
            "ordered_steps": list(self.ordered_steps),
            "layers": [layer.to_dict() for layer in self.layers],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": duration,
        }
