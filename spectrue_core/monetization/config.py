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
    """Configuration for the monetization system."""

    # Firestore paths
    balance_field: str = "balance_sc"
    available_field: str = "available_sc"
    legacy_balance_field: str = "credits"
    user_collection: str = "users"
    ledger_collection: str = "billing_ledger"
    pool_doc_path: str = "billing/free_subsidy_pool"
    bonus_state_doc_path: str = "billing/daily_bonus_state"

    # Pool bootstrap (when doc doesn't exist yet)
    initial_pool_available_sc: MoneySC = MoneySC(Decimal("1000"))
    pool_bootstrap_source: str = "bootstrap_default"

    # Pool locking
    pool_lock_days: int = 90
    pool_lock_ratio: Decimal = Decimal("0.50")  # OPERATIONAL_LOCK_RATIO
    pool_reserve_sc: MoneySC = MoneySC(Decimal("0"))

    # Parallel reservations limit
    max_parallel_reservations: int = 3

    # Active user window (days since last_seen_at)
    active_window_days: int = 6

    # Share bonus ratio (20% of daily b)
    share_ratio: Decimal = Decimal("0.20")

    # Daily bonus per-user caps
    max_bonus_per_user_sc: Decimal = Decimal("5.0")
    min_bonus_per_user_sc: Decimal = Decimal("0.0")

    # Batch processing limits
    max_users_per_batch: int = 500

    # EMA smoothing for daily budget
    ema_alpha: Decimal = Decimal("0.3")

    # Safety: max total daily budget (prevents runaway)
    max_daily_budget_sc: Decimal = Decimal("1000.0")
