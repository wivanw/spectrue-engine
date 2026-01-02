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

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectrue_core.billing.metering import LLMMeter

_current_llm_meter: ContextVar[LLMMeter | None] = ContextVar("current_llm_meter", default=None)

def get_current_llm_meter() -> LLMMeter | None:
    """Get the current LLMMeter from context."""
    return _current_llm_meter.get()

def set_current_llm_meter(meter: LLMMeter) -> None:
    """Set the current LLMMeter in context."""
    _current_llm_meter.set(meter)
