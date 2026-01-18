# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Charging Service for Monetization.

This service handles run charging with the split:
- First consume from available_sc (bonus credits)
- Then consume from balance_sc (paid credits)
- No runtime pool access
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from spectrue_core.monetization.types import (
    ChargeResult,
    ChargeSplit,
    MoneySC,
    UserWallet,
)
from spectrue_core.monetization.ledger import build_charge_idempotency_key

if TYPE_CHECKING:
    from spectrue_core.monetization.config import MonetizationConfig


class ChargingStore(Protocol):
    """Protocol for charging storage operations."""

    def read_user_wallet(self, uid: str) -> UserWallet:
        """Read user wallet."""
        ...

    def apply_charge(
        self,
        uid: str,
        run_id: str,
        split: ChargeSplit,
        idempotency_key: str,
    ) -> ChargeResult:
        """
        Apply a charge to the user's wallet.
        Deducts split.take_available from available_sc and split.take_balance from balance_sc.
        Returns ChargeResult with new balances.
        Uses idempotency_key to prevent double charging.
        """
        ...

    def check_idempotency(self, idempotency_key: str) -> ChargeResult | None:
        """Check if a charge was already applied for this idempotency key."""
        ...


@dataclass(frozen=True, slots=True)
class ChargeRequest:
    """Request to charge a user for a run."""
    uid: str
    run_id: str
    cost_sc: MoneySC


class ChargingService:
    """Service for charging users for runs using split logic."""

    def __init__(self, store: ChargingStore, cfg: "MonetizationConfig"):
        self.store = store
        self.cfg = cfg

    def compute_split(self, wallet: UserWallet, cost: MoneySC) -> ChargeSplit:
        """
        Compute how to split the charge between available and balance.

        Order: available_sc first (bonus credits), then balance_sc (paid credits).
        """
        # Take as much as possible from available_sc
        take_available = wallet.available_sc.min(cost)

        # Remaining goes to balance_sc
        remaining = cost - take_available
        take_balance = wallet.balance_sc.min(remaining)

        return ChargeSplit(
            take_available=take_available,
            take_balance=take_balance,
        )

    def can_afford(self, wallet: UserWallet, cost: MoneySC) -> bool:
        """Check if user has enough total credits to cover the cost."""
        total = wallet.available_sc + wallet.balance_sc
        return total >= cost

    def generate_idempotency_key(self, uid: str, run_id: str) -> str:
        """Generate a stable idempotency key for a charge.

        IMPORTANT:
        - Must be stable for the whole run.
        - MUST NOT include `cost_sc` (actual cost may vary slightly due to rounding,
          metering differences, or post-run accounting), otherwise retries could
          double-charge.
        """
        return build_charge_idempotency_key(uid, run_id)

    def charge(self, request: ChargeRequest) -> ChargeResult:
        """
        Charge a user for a run.

        Steps:
        1. Check idempotency (already charged?)
        2. Read wallet
        3. Compute split
        4. Verify sufficient funds
        5. Apply charge (ledger-first)
        6. Return result

        Args:
            request: ChargeRequest with uid, run_id, and cost_sc

        Returns:
            ChargeResult with ok/fail, split details, and new balances
        """
        idempotency_key = self.generate_idempotency_key(request.uid, request.run_id)



        # Read current wallet
        wallet = self.store.read_user_wallet(request.uid)

        # Check if user can afford
        if not self.can_afford(wallet, request.cost_sc):
            # Compute what we could take (partial info for error)
            split = self.compute_split(wallet, request.cost_sc)
            return ChargeResult(
                ok=False,
                split=split,
                new_balance_sc=wallet.balance_sc,
                new_available_sc=wallet.available_sc,
                reason="insufficient_funds",
            )

        # Compute the split
        split = self.compute_split(wallet, request.cost_sc)

        # Apply the charge (store handles atomicity and ledger)
        result = self.store.apply_charge(
            uid=request.uid,
            run_id=request.run_id,
            split=split,
            idempotency_key=idempotency_key,
        )

        return result

    def estimate_split(self, uid: str, cost: MoneySC) -> dict:
        """
        Estimate how a charge would be split (read-only, no mutation).

        Returns a dict with estimated split details.
        """
        wallet = self.store.read_user_wallet(uid)
        split = self.compute_split(wallet, cost)
        can_afford = self.can_afford(wallet, cost)

        return {
            "can_afford": can_afford,
            "cost_sc": cost.to_str(),
            "take_available_sc": split.take_available.to_str(),
            "take_balance_sc": split.take_balance.to_str(),
            "total_charged_sc": split.total.to_str(),
            "shortfall_sc": (cost - split.total).max0().to_str() if not can_afford else "0",
            "current_available_sc": wallet.available_sc.to_str(),
            "current_balance_sc": wallet.balance_sc.to_str(),
        }


class ChargingServiceFactory:
    """Factory for creating ChargingService instances."""

    @staticmethod
    def create(store: ChargingStore, cfg: "MonetizationConfig") -> ChargingService:
        return ChargingService(store, cfg)

