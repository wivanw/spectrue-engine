# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Daily Bonus Service for Monetization.

This service handles the daily bonus job that:
1. Unlocks matured pool buckets
2. Finds active users (seen within N days)
3. Computes budget B and per-user bonus b
4. Awards b to each active user's available_sc
5. Deducts total from pool available
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, List, Protocol, Tuple

from spectrue_core.monetization.types import (
    DailyBonusState,
    FreePool,
    LockedBucket,
    MoneySC,
    UserWallet,
)

if TYPE_CHECKING:
    from spectrue_core.monetization.config import MonetizationConfig


class DailyBonusStore(Protocol):
    """Protocol for daily bonus storage operations."""

    def list_active_users(self, today: date, window_days: int) -> List[str]:
        """List user IDs that are considered active (seen within window_days)."""
        ...

    def read_user_wallet(self, uid: str) -> UserWallet:
        """Read user wallet."""
        ...

    def read_pool(self) -> FreePool:
        """Read the free pool state."""
        ...

    def read_daily_bonus_state(self) -> DailyBonusState:
        """Read the daily bonus job state."""
        ...

    def apply_daily_bonus_batch(
        self,
        user_ids: List[str],
        bonus_per_user: MoneySC,
        today: date,
        new_pool: FreePool,
        new_state: DailyBonusState,
    ) -> int:
        """
        Atomically apply daily bonus to a batch of users.
        Returns the number of users actually awarded (skips those already awarded today).
        """
        ...


class DailyBonusService:
    """Service for computing and distributing daily bonuses."""

    def __init__(self, store: DailyBonusStore, cfg: "MonetizationConfig"):
        self.store = store
        self.cfg = cfg

    def is_active(self, last_seen_at: datetime | None, today: date) -> bool:
        """Check if a user is considered active."""
        if not last_seen_at:
            return False
        days_since = (today - last_seen_at.date()).days
        return days_since <= self.cfg.active_window_days

    def compute_b_and_B(
        self,
        pool_spendable: MoneySC,
        active_count: int,
        state: DailyBonusState,
    ) -> Tuple[MoneySC, MoneySC, DailyBonusState]:
        """
        Compute per-user bonus (b) and total budget (B) using EMA smoothing.

        Formula:
        - raw_B = min(pool_spendable, max_daily_budget)
        - ema_B = alpha * raw_B + (1 - alpha) * prev_ema_B
        - B = clamp(ema_B, 0, pool_spendable)
        - b = clamp(B / active_count, min_bonus, max_bonus)

        Returns: (b, B, new_state)
        """
        if active_count <= 0:
            # No active users, no bonus to distribute
            return MoneySC.zero(), MoneySC.zero(), state

        # Cap raw budget to safety limit
        max_budget = MoneySC(self.cfg.max_daily_budget_sc)
        raw_B = pool_spendable.min(max_budget)

        # EMA smoothing
        alpha = self.cfg.ema_alpha
        prev_ema = state.ema_budget_sc.value
        new_ema_value = alpha * raw_B.value + (Decimal("1") - alpha) * prev_ema
        ema_B = MoneySC(new_ema_value)

        # Clamp B to available pool
        B = ema_B.min(pool_spendable).max0()

        # Compute per-user bonus
        raw_b = MoneySC(B.value / active_count)
        min_b = MoneySC(self.cfg.min_bonus_per_user_sc)
        max_b = MoneySC(self.cfg.max_bonus_per_user_sc)
        b = raw_b.max(min_b).min(max_b)

        # Update state
        new_state = replace(
            state,
            ema_budget_sc=ema_B,
            active_user_count=active_count,
            last_b_sc=b,
        )

        return b, B, new_state

    def unlock_matured_buckets(self, pool: FreePool, today: date) -> FreePool:
        """Unlock any buckets that have matured (unlock_at <= today)."""
        new_locked: List[LockedBucket] = []
        unlocked_total = Decimal("0")

        for bucket in pool.locked_buckets:
            if bucket.unlock_at <= today:
                unlocked_total += bucket.amount_sc.value
            else:
                new_locked.append(bucket)

        new_available = MoneySC(pool.available_balance_sc.value + unlocked_total)
        return FreePool(
            available_balance_sc=new_available,
            locked_buckets=new_locked,
            updated_at=datetime.now(timezone.utc),
        )

    def run_daily(self, today: date | None = None) -> dict:
        """
        Execute the daily bonus job.

        Steps:
        1. Unlock matured pool buckets
        2. Find active users (seen <= active_window_days)
        3. Compute B and b
        4. Award b to each active user's available_sc
        5. Deduct total distributed from pool

        Returns a summary dict with stats.
        """
        if today is None:
            today = date.today()

        # Read current state
        pool = self.store.read_pool()
        state = self.store.read_daily_bonus_state()

        # Check if already run today
        if state.last_run_date == today:
            return {
                "status": "already_run",
                "last_run_date": today.isoformat(),
                "message": "Daily bonus job already executed today.",
            }

        # Step 1: Unlock matured buckets
        pool = self.unlock_matured_buckets(pool, today)

        # Step 2: Find active users
        active_user_ids = self.store.list_active_users(today, self.cfg.active_window_days)
        active_count = len(active_user_ids)

        if active_count == 0:
            # No active users, update state and return
            new_state = replace(state, last_run_date=today, active_user_count=0)
            # Still need to persist the pool unlock
            self.store.apply_daily_bonus_batch(
                user_ids=[],
                bonus_per_user=MoneySC.zero(),
                today=today,
                new_pool=pool,
                new_state=new_state,
            )
            return {
                "status": "no_active_users",
                "active_count": 0,
                "bonus_per_user_sc": "0",
                "total_distributed_sc": "0",
            }

        # Step 3: Compute B and b
        b, B, new_state = self.compute_b_and_B(pool.available_balance_sc, active_count, state)
        new_state = replace(new_state, last_run_date=today)

        # Step 4: Award bonuses (ALLOWANCE ONLY - NO POOL DEDUCTION)
        # We do NOT deduct from pool.available_balance_sc here.
        # Deduction happens at spend time.
        
        # However, we DO need to persist the unlocked buckets from Step 1 if any.
        # 'pool' variable already has the unlocked state from self.unlock_matured_buckets().
        # We just use that.
        new_pool = pool

        # Apply in batches
        awarded_count = 0
        for i in range(0, len(active_user_ids), self.cfg.max_users_per_batch):
            batch = active_user_ids[i : i + self.cfg.max_users_per_batch]
            awarded_count += self.store.apply_daily_bonus_batch(
                user_ids=batch,
                bonus_per_user=b,
                today=today,
                new_pool=new_pool,
                new_state=new_state,
            )

        return {
            "status": "success",
            "active_count": active_count,
            "awarded_count": awarded_count,
            "bonus_per_user_sc": b.to_str(),
            "budget_B_sc": B.to_str(),
            "total_distributed_sc": MoneySC(b.value * awarded_count).to_str(),
            "pool_available_sc": new_pool.available_balance_sc.to_str(),
            "note": "Pool is not deducted until spend.",
        }
