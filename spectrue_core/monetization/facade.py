# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from decimal import Decimal

from spectrue_core.monetization.config import MonetizationConfig
from spectrue_core.monetization.ledger import (
    LedgerEntry,
    LedgerEntryStatus,
    LedgerEntryType,
    build_idempotency_key,
)
from spectrue_core.monetization.policy import BillingPolicy
from spectrue_core.monetization.services.billing import (
    BillingStore,
    InsufficientFundsError,
    LedgerStatusError,
    ReservationLimitError,
    ReservationOutcome,
    SettlementOutcome,
)
from spectrue_core.monetization.services.free_pool import deposit as pool_deposit, deduct as pool_deduct
from spectrue_core.monetization.types import (
    MoneySC,
    PoolBalance,
    PoolStats,
    UserBalance,
    UserBalanceStats,
    quantize_sc,
)


class BillingFacade:
    def __init__(
        self,
        *,
        store: BillingStore,
        policy: BillingPolicy,
        config: MonetizationConfig | None = None,
    ) -> None:
        self._store = store
        self._policy = policy
        self._config = config or MonetizationConfig()

    def reserve_run(
        self,
        *,
        user_id: str,
        run_id: str,
        estimated_sc: MoneySC,
        meta: dict | None = None,
    ) -> ReservationOutcome:
        eligibility = self._policy.get_user_eligibility(user_id)
        outcome = self._store.reserve_run(
            user_id=user_id,
            run_id=run_id,
            estimated_sc=estimated_sc,
            eligibility=eligibility,
            max_parallel_reservations=self._config.max_parallel_reservations,
            meta=meta,
        )
        if outcome.entry.status == LedgerEntryStatus.FAILED:
            reason = outcome.entry.meta.get("reason")
            if reason == "reservation_limit":
                raise ReservationLimitError("Max parallel reservations exceeded.")
            if reason in {"insufficient_funds", "pool_insufficient"}:
                raise InsufficientFundsError("Insufficient balance or pool funds.")
            raise LedgerStatusError("Reservation failed.")
        return outcome

    def settle_run(
        self,
        *,
        user_id: str,
        run_id: str,
        actual_sc: MoneySC,
        meta: dict | None = None,
    ) -> SettlementOutcome:
        return self._store.settle_run(
            user_id=user_id,
            run_id=run_id,
            actual_sc=actual_sc,
            meta=meta,
        )

    def get_user_stats(self, user_id: str) -> UserBalanceStats:
        user_balance = self._store.get_user_balance(user_id)
        if user_balance is None:
            user_balance = UserBalance(user_id=user_id, balance_sc=Decimal("0"))
        eligibility = self._policy.get_user_eligibility(user_id)
        user_balance = UserBalance(
            user_id=user_balance.user_id,
            balance_sc=user_balance.balance_sc,
            legacy_credits=user_balance.legacy_credits,
            subsidy_allowance=eligibility.to_allowance(),
        )
        return user_balance.to_stats()

    def get_pool_stats(self) -> PoolStats:
        pool_balance = self._store.get_pool_balance()
        if pool_balance.reserve_sc != self._config.pool_reserve_sc:
            pool_balance = PoolBalance(
                available_balance_sc=pool_balance.available_balance_sc,
                locked_buckets=pool_balance.locked_buckets,
                reserve_sc=self._config.pool_reserve_sc,
            )
        return pool_balance.to_stats()

    def credit_user(
        self,
        *,
        user_id: str,
        amount_sc: MoneySC,
        event_id: str | None = None,
        idempotency_key: str | None = None,
        meta: dict | None = None,
    ) -> LedgerEntry | None:
        if amount_sc <= 0:
            return None
        key = idempotency_key or build_idempotency_key("user", user_id, "credit", event_id or "")
        existing = self._store.get_ledger_entry(key)
        if existing:
            return existing
        user_balance = self._store.get_user_balance(user_id)
        if user_balance is None:
            user_balance = UserBalance(user_id=user_id, balance_sc=Decimal("0"))
        new_balance = quantize_sc(user_balance.balance_sc + amount_sc)
        self._store.set_user_balance(user_id, new_balance)
        entry = LedgerEntry(
            idempotency_key=key,
            entry_type=LedgerEntryType.DEPOSIT,
            amount_sc=amount_sc,
            status=LedgerEntryStatus.SETTLED,
            event_id=event_id,
            user_id=user_id,
            meta=meta or {},
        )
        self._store.write_ledger_entry(entry)
        return entry

    def deposit_pool(
        self,
        *,
        amount_sc: MoneySC,
        event_id: str | None = None,
        idempotency_key: str | None = None,
        meta: dict | None = None,
    ) -> LedgerEntry | None:
        if amount_sc <= 0:
            return None
        key = idempotency_key or build_idempotency_key("pool", "deposit", event_id or "")
        existing = self._store.get_ledger_entry(key)
        if existing:
            return existing
        pool_balance = self._store.get_pool_balance()
        updated_pool = pool_deposit(
            pool_balance,
            amount_sc,
            lock_ratio=self._config.pool_lock_ratio,
            lock_days=self._config.pool_lock_days,
        )
        self._store.set_pool_balance(updated_pool)
        entry = LedgerEntry(
            idempotency_key=key,
            entry_type=LedgerEntryType.DEPOSIT,
            amount_sc=amount_sc,
            status=LedgerEntryStatus.SETTLED,
            event_id=event_id,
            meta=meta or {},
        )
        self._store.write_ledger_entry(entry)
        return entry

    def spend_pool(
        self,
        *,
        amount_sc: MoneySC,
        event_id: str | None = None,
        idempotency_key: str | None = None,
        meta: dict | None = None,
    ) -> bool:
        if amount_sc <= 0:
            return True
        key = idempotency_key or build_idempotency_key("pool", "spend", event_id or "")
        existing = self._store.get_ledger_entry(key)
        if existing:
            return existing.status == LedgerEntryStatus.SETTLED
        pool_balance = self._store.get_pool_balance()
        updated_pool, ok = pool_deduct(pool_balance, amount_sc)
        if not ok:
            return False
        self._store.set_pool_balance(updated_pool)
        entry = LedgerEntry(
            idempotency_key=key,
            entry_type=LedgerEntryType.ADJUSTMENT,
            amount_sc=amount_sc,
            status=LedgerEntryStatus.SETTLED,
            event_id=event_id,
            meta=meta or {},
        )
        self._store.write_ledger_entry(entry)
        return True

    def record_profit(
        self,
        *,
        amount_sc: MoneySC,
        event_id: str | None = None,
        idempotency_key: str | None = None,
        meta: dict | None = None,
    ) -> LedgerEntry | None:
        if amount_sc <= 0:
            return None
        key = idempotency_key or build_idempotency_key("profit", event_id or "")
        existing = self._store.get_ledger_entry(key)
        if existing:
            return existing
        entry = LedgerEntry(
            idempotency_key=key,
            entry_type=LedgerEntryType.ADJUSTMENT,
            amount_sc=amount_sc,
            status=LedgerEntryStatus.SETTLED,
            event_id=event_id,
            meta=meta or {},
        )
        self._store.write_ledger_entry(entry)
        return entry
