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
Pipeline Errors

Custom exceptions for pipeline execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineViolation(Exception):
    """
    Raised when a pipeline invariant is violated.

    This exception indicates a structural problem with the input
    that violates the pipeline mode's requirements (e.g., multi-claim
    input in normal mode).

    Attributes:
        step_name: Name of the step that detected the violation
        invariant: Description of the violated invariant
        expected: What was expected
        actual: What was actually found
        details: Additional context for debugging
    """

    step_name: str
    invariant: str
    expected: Any
    actual: Any
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        msg = (
            f"Pipeline violation in '{self.step_name}': {self.invariant}. "
            f"Expected: {self.expected}, Actual: {self.actual}"
        )
        super().__init__(msg)

    def to_trace_dict(self) -> dict[str, Any]:
        """Convert to dictionary for trace logging."""
        return {
            "error": "pipeline_violation",
            "step_name": self.step_name,
            "invariant": self.invariant,
            "expected": str(self.expected),
            "actual": str(self.actual),
            "details": self.details,
        }


class PipelineExecutionError(Exception):
    """
    Raised when pipeline execution fails for non-invariant reasons.

    This covers runtime errors like network failures, LLM timeouts,
    or unexpected exceptions during step execution.
    """

    def __init__(self, step_name: str, message: str, cause: Exception | None = None):
        self.step_name = step_name
        self.cause = cause
        full_msg = f"Pipeline execution failed at '{step_name}': {message}"
        if cause:
            full_msg += f" (caused by: {cause})"
        super().__init__(full_msg)
