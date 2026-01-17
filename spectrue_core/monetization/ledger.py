# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from spectrue_core.monetization.types import MoneySC


class LedgerEntryType(str, Enum):
    RESERVE = "reserve"
    SETTLE = "settle"
    RELEASE = "release"
    DEPOSIT = "deposit"
    ADJUSTMENT = "adjustment"


class LedgerEntryStatus(str, Enum):
    PENDING = "pending"
    SETTLED = "settled"
    REVERSED = "reversed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    idempotency_key: str
    entry_type: LedgerEntryType
    amount_sc: MoneySC
    status: LedgerEntryStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    event_id: str | None = None
    user_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def build_idempotency_key(*parts: str) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return ":".join(cleaned)
