# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from spectrue_core.monetization.types import MoneySC


@dataclass(frozen=True, slots=True)
class MonetizationConfig:
    balance_field: str = "balance_sc"
    legacy_balance_field: str = "credits"
    user_collection: str = "users"
    ledger_collection: str = "billing_ledger"
    pool_doc_path: str = "system_stats/monetization"
    pool_lock_days: int = 90
    pool_lock_ratio: Decimal = Decimal("0.0")
    pool_reserve_sc: MoneySC = Decimal("0")
    max_parallel_reservations: int = 3
