# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for DailyBonusService."""

from datetime import date, datetime
from decimal import Decimal
from typing import List


from spectrue_core.monetization.config import MonetizationConfig
from spectrue_core.monetization.services.daily_bonus import DailyBonusService
from spectrue_core.monetization.types import (
    DailyBonusState,
    FreePool,
    MoneySC,
    UserWallet,
)


class MockDailyBonusStore:
    """In-memory mock for DailyBonusStore protocol."""

    def __init__(self):
        self.users: dict[str, UserWallet] = {}
        self.pool = FreePool(available_balance_sc=MoneySC(Decimal("1000")))
        self.state = DailyBonusState(ema_budget_sc=MoneySC(Decimal("100")))
        self.awarded_users: List[str] = []

    def list_active_users(self, today: date, window_days: int) -> List[str]:
        result = []
        for uid, wallet in self.users.items():
            if wallet.last_seen_at:
                days_since = (today - wallet.last_seen_at.date()).days
                if days_since <= window_days:
                    result.append(uid)
        return result

    def read_user_wallet(self, uid: str) -> UserWallet:
        if uid in self.users:
            return self.users[uid]
        return UserWallet(
            uid=uid,
            balance_sc=MoneySC.zero(),
            available_sc=MoneySC.zero(),
        )

    def read_pool(self) -> FreePool:
        return self.pool

    def read_daily_bonus_state(self) -> DailyBonusState:
        return self.state

    def apply_daily_bonus_batch(
        self,
        user_ids: List[str],
        bonus_per_user: MoneySC,
        today: date,
        new_pool: FreePool,
        new_state: DailyBonusState,
    ) -> int:
        awarded = 0
        for uid in user_ids:
            wallet = self.users.get(uid)
            if wallet and wallet.last_daily_bonus_date != today:
                # Award bonus
                self.users[uid] = UserWallet(
                    uid=uid,
                    balance_sc=wallet.balance_sc,
                    available_sc=wallet.available_sc + bonus_per_user,
                    last_seen_at=wallet.last_seen_at,
                    last_daily_bonus_date=today,
                    last_share_bonus_date=wallet.last_share_bonus_date,
                )
                awarded += 1
                self.awarded_users.append(uid)

        self.pool = new_pool
        self.state = new_state
        return awarded


class TestDailyBonusServiceIsActive:
    """Tests for is_active method."""

    def test_active_within_window(self):
        cfg = MonetizationConfig(active_window_days=6)
        store = MockDailyBonusStore()
        service = DailyBonusService(store, cfg)

        today = date(2026, 1, 17)
        last_seen = datetime(2026, 1, 12, 10, 0)  # 5 days ago
        assert service.is_active(last_seen, today) is True

    def test_inactive_outside_window(self):
        cfg = MonetizationConfig(active_window_days=6)
        store = MockDailyBonusStore()
        service = DailyBonusService(store, cfg)

        today = date(2026, 1, 17)
        last_seen = datetime(2026, 1, 10, 10, 0)  # 7 days ago
        assert service.is_active(last_seen, today) is False

    def test_inactive_when_none(self):
        cfg = MonetizationConfig(active_window_days=6)
        store = MockDailyBonusStore()
        service = DailyBonusService(store, cfg)

        today = date(2026, 1, 17)
        assert service.is_active(None, today) is False


class TestDailyBonusServiceComputeBandB:
    """Tests for compute_b_and_B method."""

    def test_compute_with_active_users(self):
        cfg = MonetizationConfig(
            max_bonus_per_user_sc=Decimal("5.0"),
            min_bonus_per_user_sc=Decimal("0.0"),
            max_daily_budget_sc=Decimal("1000.0"),
            ema_alpha=Decimal("0.3"),
        )
        store = MockDailyBonusStore()
        service = DailyBonusService(store, cfg)

        pool_spendable = MoneySC(Decimal("100"))
        active_count = 10
        state = DailyBonusState(ema_budget_sc=MoneySC(Decimal("50")))

        b, B, new_state = service.compute_b_and_B(pool_spendable, active_count, state)

        # EMA: 0.3 * 100 + 0.7 * 50 = 30 + 35 = 65
        assert new_state.ema_budget_sc.value == Decimal("65")
        # B = min(65, 100) = 65
        # b = 65 / 10 = 6.5 -> capped at max_bonus = 5.0
        assert b.value == Decimal("5.0")

    def test_compute_no_active_users(self):
        cfg = MonetizationConfig()
        store = MockDailyBonusStore()
        service = DailyBonusService(store, cfg)

        pool_spendable = MoneySC(Decimal("100"))
        active_count = 0
        state = DailyBonusState(ema_budget_sc=MoneySC(Decimal("50")))

        b, B, new_state = service.compute_b_and_B(pool_spendable, active_count, state)

        assert b.value == Decimal("0")
        assert B.value == Decimal("0")


class TestDailyBonusServiceUnlockBuckets:
    """Tests for unlock_matured_buckets method."""

    def test_unlock_matured_bucket(self):
        cfg = MonetizationConfig()
        store = MockDailyBonusStore()
        service = DailyBonusService(store, cfg)

        from spectrue_core.monetization.types import LockedBucket

        pool = FreePool(
            available_balance_sc=MoneySC(Decimal("500")),
            locked_buckets=[
                LockedBucket(MoneySC(Decimal("100")), date(2026, 1, 15)),  # Matured
                LockedBucket(MoneySC(Decimal("200")), date(2026, 4, 17)),  # Not matured
            ],
        )

        today = date(2026, 1, 17)
        new_pool = service.unlock_matured_buckets(pool, today)

        # 100 should be unlocked
        assert new_pool.available_balance_sc.value == Decimal("600")
        assert len(new_pool.locked_buckets) == 1
        assert new_pool.locked_buckets[0].amount_sc.value == Decimal("200")


class TestDailyBonusServiceRunDaily:
    """Tests for run_daily method."""

    def test_run_daily_awards_bonus(self):
        cfg = MonetizationConfig(
            active_window_days=6,
            max_bonus_per_user_sc=Decimal("5.0"),
        )
        store = MockDailyBonusStore()
        store.pool = FreePool(available_balance_sc=MoneySC(Decimal("1000")))

        # Add active users
        today = date(2026, 1, 17)
        store.users = {
            "user1": UserWallet(
                uid="user1",
                balance_sc=MoneySC(Decimal("100")),
                available_sc=MoneySC(Decimal("0")),
                last_seen_at=datetime(2026, 1, 16, 10, 0),
            ),
            "user2": UserWallet(
                uid="user2",
                balance_sc=MoneySC(Decimal("50")),
                available_sc=MoneySC(Decimal("10")),
                last_seen_at=datetime(2026, 1, 15, 10, 0),
            ),
        }

        service = DailyBonusService(store, cfg)
        result = service.run_daily(today)

        assert result["status"] == "success"
        assert result["active_count"] == 2
        assert result["awarded_count"] == 2
        assert len(store.awarded_users) == 2

    def test_run_daily_already_run_today(self):
        cfg = MonetizationConfig()
        store = MockDailyBonusStore()
        today = date(2026, 1, 17)
        store.state = DailyBonusState(
            ema_budget_sc=MoneySC(Decimal("100")),
            last_run_date=today,  # Already run today
        )

        service = DailyBonusService(store, cfg)
        result = service.run_daily(today)

        assert result["status"] == "already_run"

    def test_run_daily_no_active_users(self):
        cfg = MonetizationConfig(active_window_days=6)
        store = MockDailyBonusStore()
        store.users = {}  # No users

        service = DailyBonusService(store, cfg)
        result = service.run_daily(date(2026, 1, 17))

        assert result["status"] == "no_active_users"
        assert result["active_count"] == 0
