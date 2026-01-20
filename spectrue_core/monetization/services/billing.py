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
from typing import Protocol, runtime_checkable

from spectrue_core.monetization.ledger import LedgerEntry, LedgerEntryStatus
from spectrue_core.monetization.policy import UserEligibility
from spectrue_core.monetization.types import MoneySC, PoolBalance, UserBalance


class BillingError(RuntimeError):
    pass


class IdempotencyError(BillingError):
    pass


class InsufficientFundsError(BillingError):
    pass


class ReservationLimitError(BillingError):
    pass


class LedgerStatusError(BillingError):
    pass


@dataclass(frozen=True, slots=True)
class ReservationOutcome:
    entry: LedgerEntry
    user_balance: UserBalance
    pool_balance: PoolBalance
    reserved_user_sc: MoneySC
    reserved_pool_sc: MoneySC


@dataclass(frozen=True, slots=True)
class SettlementOutcome:
    entry: LedgerEntry
    user_balance: UserBalance
    pool_balance: PoolBalance
    charged_user_sc: MoneySC
    charged_pool_sc: MoneySC
    refunded_user_sc: MoneySC
    refunded_pool_sc: MoneySC


@runtime_checkable
class BillingStore(Protocol):
    def get_user_balance(self, user_id: str) -> UserBalance | None:
        ...

    def set_user_balance(self, user_id: str, balance_sc: MoneySC) -> UserBalance:
        ...

    def get_pool_balance(self) -> PoolBalance:
        ...

    def set_pool_balance(self, pool: PoolBalance) -> None:
        ...

    def get_ledger_entry(self, idempotency_key: str) -> LedgerEntry | None:
        ...

    def write_ledger_entry(self, entry: LedgerEntry) -> LedgerEntry:
        ...

    def update_ledger_status(
        self,
        idempotency_key: str,
        status: LedgerEntryStatus,
        *,
        meta: dict | None = None,
    ) -> LedgerEntry:
        ...

    def count_open_reservations(self, user_id: str) -> int:
        ...

    def reserve_run(
        self,
        *,
        user_id: str,
        run_id: str,
        estimated_sc: MoneySC,
        eligibility: UserEligibility,
        max_parallel_reservations: int,
        meta: dict | None = None,
    ) -> ReservationOutcome:
        ...

    def settle_run(
        self,
        *,
        user_id: str,
        run_id: str,
        actual_sc: MoneySC,
        meta: dict | None = None,
    ) -> SettlementOutcome:
        ...


def validate_settle_entry(entry: LedgerEntry) -> bool:
    if entry.status == LedgerEntryStatus.SETTLED:
        return True
    if entry.status == LedgerEntryStatus.PENDING:
        return False
    raise LedgerStatusError("Reservation cannot be settled from current status.")
