# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""LLM utilities package."""

from .failures import (
    LLMFailureKind,
    classify_llm_failure,
    is_fallback_eligible,
    failure_kind_to_trace_data,
)

__all__ = [
    "LLMFailureKind",
    "classify_llm_failure",
    "is_fallback_eligible",
    "failure_kind_to_trace_data",
]
