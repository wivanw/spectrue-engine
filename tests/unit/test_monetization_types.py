# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for Monetization types."""

from datetime import date
from decimal import Decimal

import pytest

from spectrue_core.monetization.types import (
    ChargeResult,
    ChargeSplit,
    DailyBonusState,
    FreePool,
    LockedBucket,
    MoneySC,
    UserWallet,
)


class TestMoneySC:
    """Tests for MoneySC dataclass."""

    def test_creation_quantizes(self):
        """MoneySC should quantize on creation."""
        m = MoneySC(Decimal("1.123456789"))
        assert m.value == Decimal("1.123457")

    def test_addition(self):
        a = MoneySC(Decimal("1.5"))
        b = MoneySC(Decimal("2.3"))
        result = a + b
        assert result.value == Decimal("3.8")

    def test_subtraction(self):
        a = MoneySC(Decimal("5.0"))
        b = MoneySC(Decimal("2.0"))
        result = a - b
        assert result.value == Decimal("3.0")

    def test_multiplication(self):
        m = MoneySC(Decimal("2.0"))
        result = m * 3
        assert result.value == Decimal("6.0")

    def test_division(self):
        m = MoneySC(Decimal("10.0"))
        result = m / 4
        assert result.value == Decimal("2.5")

    def test_division_by_zero_raises(self):
        m = MoneySC(Decimal("10.0"))
        with pytest.raises(ZeroDivisionError):
            _ = m / 0

    def test_comparison_operators(self):
        a = MoneySC(Decimal("1.0"))
        b = MoneySC(Decimal("2.0"))
        assert a < b
        assert a <= b
        assert b > a
        assert b >= a
        assert a <= a
        assert a >= a

    def test_min(self):
        a = MoneySC(Decimal("5.0"))
        b = MoneySC(Decimal("3.0"))
        assert a.min(b).value == Decimal("3.0")

    def test_max(self):
        a = MoneySC(Decimal("5.0"))
        b = MoneySC(Decimal("3.0"))
        assert a.max(b).value == Decimal("5.0")

    def test_max0(self):
        m = MoneySC(Decimal("-5.0"))
        assert m.max0().value == Decimal("0")

        m2 = MoneySC(Decimal("5.0"))
        assert m2.max0().value == Decimal("5.0")

    def test_to_str_integer(self):
        m = MoneySC(Decimal("5.0"))
        assert m.to_str() == "5"

    def test_to_str_decimal(self):
        m = MoneySC(Decimal("5.123"))
        assert m.to_str() == "5.123"

    def test_zero(self):
        z = MoneySC.zero()
        assert z.value == Decimal("0")

    def test_from_str(self):
        m = MoneySC.from_str("12.345")
        assert m.value == Decimal("12.345")


class TestUserWallet:
    """Tests for UserWallet dataclass."""

    def test_creation(self):
        wallet = UserWallet(
            uid="user123",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
        )
        assert wallet.uid == "user123"
        assert wallet.balance_sc.value == Decimal("100")
        assert wallet.available_sc.value == Decimal("50")

    def test_total_sc(self):
        wallet = UserWallet(
            uid="user123",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
        )
        assert wallet.total_sc().value == Decimal("150")

    def test_to_dict(self):
        wallet = UserWallet(
            uid="user123",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
            last_daily_bonus_date=date(2026, 1, 15),
        )
        d = wallet.to_dict()
        assert d["uid"] == "user123"
        assert d["balance_sc"] == "100"
        assert d["available_sc"] == "50"
        assert d["last_daily_bonus_date"] == "2026-01-15"


class TestChargeSplit:
    """Tests for ChargeSplit dataclass."""

    def test_total(self):
        split = ChargeSplit(
            take_available=MoneySC(Decimal("3")),
            take_balance=MoneySC(Decimal("5")),
        )
        assert split.total.value == Decimal("8")


class TestChargeResult:
    """Tests for ChargeResult dataclass."""

    def test_to_dict(self):
        split = ChargeSplit(
            take_available=MoneySC(Decimal("3")),
            take_balance=MoneySC(Decimal("5")),
        )
        result = ChargeResult(
            ok=True,
            split=split,
            new_balance_sc=MoneySC(Decimal("95")),
            new_available_sc=MoneySC(Decimal("47")),
        )
        d = result.to_dict()
        assert d["ok"] is True
        assert d["take_available"] == "3"
        assert d["take_balance"] == "5"
        assert d["total_charged"] == "8"


class TestLockedBucket:
    """Tests for LockedBucket dataclass."""

    def test_to_dict(self):
        bucket = LockedBucket(
            amount_sc=MoneySC(Decimal("100")),
            unlock_at=date(2026, 4, 17),
        )
        d = bucket.to_dict()
        assert d["amount_sc"] == "100"
        assert d["unlock_at"] == "2026-04-17"

    def test_from_dict(self):
        d = {"amount_sc": "100", "unlock_at": "2026-04-17"}
        bucket = LockedBucket.from_dict(d)
        assert bucket.amount_sc.value == Decimal("100")
        assert bucket.unlock_at == date(2026, 4, 17)


class TestFreePool:
    """Tests for FreePool dataclass."""

    def test_locked_total(self):
        pool = FreePool(
            available_balance_sc=MoneySC(Decimal("500")),
            locked_buckets=[
                LockedBucket(MoneySC(Decimal("100")), date(2026, 4, 17)),
                LockedBucket(MoneySC(Decimal("200")), date(2026, 5, 17)),
            ],
        )
        assert pool.locked_total().value == Decimal("300")

    def test_total(self):
        pool = FreePool(
            available_balance_sc=MoneySC(Decimal("500")),
            locked_buckets=[
                LockedBucket(MoneySC(Decimal("100")), date(2026, 4, 17)),
            ],
        )
        assert pool.total().value == Decimal("600")

    def test_to_dict(self):
        pool = FreePool(
            available_balance_sc=MoneySC(Decimal("500")),
            locked_buckets=[
                LockedBucket(MoneySC(Decimal("100")), date(2026, 4, 17)),
            ],
        )
        d = pool.to_dict()
        assert d["available_balance_sc"] == "500"
        assert d["locked_total_sc"] == "100"
        assert d["total_sc"] == "600"


class TestDailyBonusState:
    """Tests for DailyBonusState dataclass."""

    def test_to_dict(self):
        state = DailyBonusState(
            ema_budget_sc=MoneySC(Decimal("50")),
            last_run_date=date(2026, 1, 16),
            active_user_count=100,
            last_b_sc=MoneySC(Decimal("0.5")),
        )
        d = state.to_dict()
        assert d["ema_budget_sc"] == "50"
        assert d["last_run_date"] == "2026-01-16"
        assert d["active_user_count"] == 100

    def test_from_dict(self):
        d = {
            "ema_budget_sc": "50",
            "last_run_date": "2026-01-16",
            "smoothing_alpha": "0.3",
            "active_user_count": 100,
            "last_b_sc": "0.5",
        }
        state = DailyBonusState.from_dict(d)
        assert state.ema_budget_sc.value == Decimal("50")
        assert state.last_run_date == date(2026, 1, 16)
        assert state.active_user_count == 100
