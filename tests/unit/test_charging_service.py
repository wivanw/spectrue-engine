# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for ChargingService."""

from decimal import Decimal


from spectrue_core.monetization.config import MonetizationConfig
from spectrue_core.monetization.services.charging import ChargingService, ChargeRequest
from spectrue_core.monetization.types import (
    ChargeResult,
    ChargeSplit,
    MoneySC,
    UserWallet,
)


class MockChargingStore:
    """In-memory mock for ChargingStore protocol."""

    def __init__(self):
        self.wallets: dict[str, UserWallet] = {}
        self.charges: dict[str, ChargeResult] = {}

    def read_user_wallet(self, uid: str) -> UserWallet:
        if uid in self.wallets:
            return self.wallets[uid]
        return UserWallet(
            uid=uid,
            balance_sc=MoneySC.zero(),
            available_sc=MoneySC.zero(),
        )

    def check_idempotency(self, idempotency_key: str) -> ChargeResult | None:
        return self.charges.get(idempotency_key)

    def apply_charge(
        self,
        uid: str,
        run_id: str,
        split: ChargeSplit,
        idempotency_key: str,
    ) -> ChargeResult:
        wallet = self.wallets[uid]
        new_available = wallet.available_sc - split.take_available
        new_balance = wallet.balance_sc - split.take_balance

        self.wallets[uid] = UserWallet(
            uid=uid,
            balance_sc=new_balance,
            available_sc=new_available,
            last_seen_at=wallet.last_seen_at,
        )

        result = ChargeResult(
            ok=True,
            split=split,
            new_balance_sc=new_balance,
            new_available_sc=new_available,
        )
        self.charges[idempotency_key] = result
        return result


class TestChargingServiceComputeSplit:
    """Tests for compute_split method."""

    def test_split_from_available_only(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        service = ChargingService(store, cfg)

        wallet = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
        )
        cost = MoneySC(Decimal("30"))

        split = service.compute_split(wallet, cost)

        assert split.take_available.value == Decimal("30")
        assert split.take_balance.value == Decimal("0")
        assert split.total.value == Decimal("30")

    def test_split_from_both(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        service = ChargingService(store, cfg)

        wallet = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("20")),
        )
        cost = MoneySC(Decimal("50"))

        split = service.compute_split(wallet, cost)

        assert split.take_available.value == Decimal("20")
        assert split.take_balance.value == Decimal("30")
        assert split.total.value == Decimal("50")

    def test_split_from_balance_only(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        service = ChargingService(store, cfg)

        wallet = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("0")),
        )
        cost = MoneySC(Decimal("50"))

        split = service.compute_split(wallet, cost)

        assert split.take_available.value == Decimal("0")
        assert split.take_balance.value == Decimal("50")

    def test_split_exceeds_total(self):
        """When cost exceeds total, split takes what's available."""
        cfg = MonetizationConfig()
        store = MockChargingStore()
        service = ChargingService(store, cfg)

        wallet = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("30")),
            available_sc=MoneySC(Decimal("20")),
        )
        cost = MoneySC(Decimal("100"))

        split = service.compute_split(wallet, cost)

        assert split.take_available.value == Decimal("20")
        assert split.take_balance.value == Decimal("30")
        assert split.total.value == Decimal("50")  # Only 50 available


class TestChargingServiceCanAfford:
    """Tests for can_afford method."""

    def test_can_afford_true(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        service = ChargingService(store, cfg)

        wallet = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
        )
        cost = MoneySC(Decimal("150"))

        assert service.can_afford(wallet, cost) is True

    def test_can_afford_false(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        service = ChargingService(store, cfg)

        wallet = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
        )
        cost = MoneySC(Decimal("200"))

        assert service.can_afford(wallet, cost) is False


class TestChargingServiceCharge:
    """Tests for charge method."""

    def test_charge_success(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        store.wallets["user1"] = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
        )

        service = ChargingService(store, cfg)
        request = ChargeRequest(
            uid="user1",
            run_id="run123",
            cost_sc=MoneySC(Decimal("70")),
        )

        result = service.charge(request)

        assert result.ok is True
        assert result.split.take_available.value == Decimal("50")
        assert result.split.take_balance.value == Decimal("20")
        assert result.new_available_sc.value == Decimal("0")
        assert result.new_balance_sc.value == Decimal("80")

    def test_charge_insufficient_funds(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        store.wallets["user1"] = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("30")),
            available_sc=MoneySC(Decimal("10")),
        )

        service = ChargingService(store, cfg)
        request = ChargeRequest(
            uid="user1",
            run_id="run123",
            cost_sc=MoneySC(Decimal("100")),
        )

        result = service.charge(request)

        assert result.ok is False
        assert result.reason == "insufficient_funds"

    def test_charge_idempotent(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        store.wallets["user1"] = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
        )

        service = ChargingService(store, cfg)
        request = ChargeRequest(
            uid="user1",
            run_id="run123",
            cost_sc=MoneySC(Decimal("30")),
        )

        result1 = service.charge(request)
        result2 = service.charge(request)

        # Both should return same result
        assert result1.ok is True
        assert result2.ok is True
        assert result1.split.total.value == result2.split.total.value

        # Balance should only be deducted once
        wallet = store.wallets["user1"]
        assert wallet.available_sc.value == Decimal("20")  # 50 - 30


class TestChargingServiceEstimateSplit:
    """Tests for estimate_split method."""

    def test_estimate_split_can_afford(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        store.wallets["user1"] = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("100")),
            available_sc=MoneySC(Decimal("50")),
        )

        service = ChargingService(store, cfg)
        result = service.estimate_split("user1", MoneySC(Decimal("70")))

        assert result["can_afford"] is True
        assert result["take_available_sc"] == "50"
        assert result["take_balance_sc"] == "20"
        assert result["shortfall_sc"] == "0"

    def test_estimate_split_cannot_afford(self):
        cfg = MonetizationConfig()
        store = MockChargingStore()
        store.wallets["user1"] = UserWallet(
            uid="user1",
            balance_sc=MoneySC(Decimal("30")),
            available_sc=MoneySC(Decimal("10")),
        )

        service = ChargingService(store, cfg)
        result = service.estimate_split("user1", MoneySC(Decimal("100")))

        assert result["can_afford"] is False
        assert result["shortfall_sc"] == "60"  # 100 - 40 = 60
