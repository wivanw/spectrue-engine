# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Dict

from spectrue_core.monetization.ledger import (
    LedgerEntry,
    LedgerEntryStatus,
    LedgerEntryType,
    build_idempotency_key,
)
from spectrue_core.monetization.policy import UserEligibility
from spectrue_core.monetization.services.billing import (
    BillingStore,
    IdempotencyError,
    InsufficientFundsError,
    ReservationOutcome,
    SettlementOutcome,
    validate_settle_entry,
)
from spectrue_core.monetization.types import MoneySC, PoolBalance, UserBalance, quantize_sc


class InMemoryBillingStore(BillingStore):
    def __init__(self, *, pool_balance: PoolBalance | None = None) -> None:
        self._user_balances: Dict[str, UserBalance] = {}
        self._ledger_entries: Dict[str, LedgerEntry] = {}
        self._pool_balance = pool_balance or PoolBalance(
            available_balance_sc=Decimal("0"),
            locked_buckets={},
            reserve_sc=Decimal("0"),
        )

    def get_user_balance(self, user_id: str) -> UserBalance | None:
        return self._user_balances.get(user_id)

    def set_user_balance(self, user_id: str, balance_sc: MoneySC) -> UserBalance:
        existing = self._user_balances.get(user_id)
        if existing is None:
            updated = UserBalance(user_id=user_id, balance_sc=balance_sc)
        else:
            updated = UserBalance(
                user_id=user_id,
                balance_sc=balance_sc,
                legacy_credits=existing.legacy_credits,
                subsidy_allowance=existing.subsidy_allowance,
            )
        self._user_balances[user_id] = updated
        return updated

    def get_pool_balance(self) -> PoolBalance:
        return self._pool_balance

    def set_pool_balance(self, pool: PoolBalance) -> None:
        self._pool_balance = pool

    def get_ledger_entry(self, idempotency_key: str) -> LedgerEntry | None:
        return self._ledger_entries.get(idempotency_key)

    def write_ledger_entry(self, entry: LedgerEntry) -> LedgerEntry:
        existing = self._ledger_entries.get(entry.idempotency_key)
        if existing is not None:
            if existing == entry:
                return existing
            raise IdempotencyError("Ledger entry already exists for idempotency key.")
        self._ledger_entries[entry.idempotency_key] = entry
        return entry

    def update_ledger_status(
        self,
        idempotency_key: str,
        status: LedgerEntryStatus,
        *,
        meta: dict | None = None,
    ) -> LedgerEntry:
        existing = self._ledger_entries.get(idempotency_key)
        if existing is None:
            raise IdempotencyError("Ledger entry not found for idempotency key.")
        merged_meta = dict(existing.meta)
        if meta:
            merged_meta.update(meta)
        updated = replace(existing, status=status, meta=merged_meta)
        self._ledger_entries[idempotency_key] = updated
        return updated

    def count_open_reservations(self, user_id: str) -> int:
        return sum(
            1
            for entry in self._ledger_entries.values()
            if entry.user_id == user_id
            and entry.entry_type == LedgerEntryType.RESERVE
            and entry.status == LedgerEntryStatus.PENDING
        )

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
        key = build_idempotency_key("run", run_id, "reserve")
        existing = self.get_ledger_entry(key)
        if existing:
            reserved_user = Decimal(str(existing.meta.get("reserved_user_sc", "0")))
            reserved_pool = Decimal(str(existing.meta.get("reserved_pool_sc", "0")))
            user_balance = self.get_user_balance(user_id) or UserBalance(user_id=user_id, balance_sc=Decimal("0"))
            return ReservationOutcome(
                entry=existing,
                user_balance=user_balance,
                pool_balance=self.get_pool_balance(),
                reserved_user_sc=reserved_user,
                reserved_pool_sc=reserved_pool,
            )

        if self.count_open_reservations(user_id) >= max_parallel_reservations:
            entry = LedgerEntry(
                idempotency_key=key,
                entry_type=LedgerEntryType.RESERVE,
                amount_sc=estimated_sc,
                status=LedgerEntryStatus.FAILED,
                user_id=user_id,
                meta={"reason": "reservation_limit", **(meta or {})},
            )
            self.write_ledger_entry(entry)
            return ReservationOutcome(
                entry=entry,
                user_balance=self.get_user_balance(user_id) or UserBalance(user_id=user_id, balance_sc=Decimal("0")),
                pool_balance=self.get_pool_balance(),
                reserved_user_sc=Decimal("0"),
                reserved_pool_sc=Decimal("0"),
            )

        user_balance = self.get_user_balance(user_id) or UserBalance(user_id=user_id, balance_sc=Decimal("0"))
        pool_balance = self.get_pool_balance()
        eligible_cap = Decimal("0")
        if eligibility.eligible:
            eligible_cap = min(eligibility.daily_remaining_sc, eligibility.monthly_remaining_sc)
        spendable_pool = min(pool_balance.spendable_sc, eligible_cap)

        reserved_user = min(user_balance.balance_sc, estimated_sc)
        remaining = quantize_sc(estimated_sc - reserved_user)
        reserved_pool = min(spendable_pool, remaining)
        reserved_total = reserved_user + reserved_pool

        if reserved_total < estimated_sc:
            entry = LedgerEntry(
                idempotency_key=key,
                entry_type=LedgerEntryType.RESERVE,
                amount_sc=estimated_sc,
                status=LedgerEntryStatus.FAILED,
                user_id=user_id,
                meta={
                    "reason": "insufficient_funds",
                    "reserved_user_sc": str(reserved_user),
                    "reserved_pool_sc": str(reserved_pool),
                    **(meta or {}),
                },
            )
            self.write_ledger_entry(entry)
            return ReservationOutcome(
                entry=entry,
                user_balance=user_balance,
                pool_balance=pool_balance,
                reserved_user_sc=reserved_user,
                reserved_pool_sc=reserved_pool,
            )

        new_user_balance = UserBalance(
            user_id=user_id,
            balance_sc=quantize_sc(user_balance.balance_sc - reserved_user),
            legacy_credits=user_balance.legacy_credits,
            subsidy_allowance=user_balance.subsidy_allowance,
        )
        new_pool_balance = PoolBalance(
            available_balance_sc=quantize_sc(pool_balance.available_balance_sc - reserved_pool),
            locked_buckets=dict(pool_balance.locked_buckets),
            reserve_sc=pool_balance.reserve_sc,
        )

        entry = LedgerEntry(
            idempotency_key=key,
            entry_type=LedgerEntryType.RESERVE,
            amount_sc=estimated_sc,
            status=LedgerEntryStatus.PENDING,
            user_id=user_id,
            meta={
                "reserved_user_sc": str(reserved_user),
                "reserved_pool_sc": str(reserved_pool),
                **(meta or {}),
            },
        )
        self.write_ledger_entry(entry)
        self._user_balances[user_id] = new_user_balance
        self._pool_balance = new_pool_balance
        return ReservationOutcome(
            entry=entry,
            user_balance=new_user_balance,
            pool_balance=new_pool_balance,
            reserved_user_sc=reserved_user,
            reserved_pool_sc=reserved_pool,
        )

    def settle_run(
        self,
        *,
        user_id: str,
        run_id: str,
        actual_sc: MoneySC,
        meta: dict | None = None,
    ) -> SettlementOutcome:
        key = build_idempotency_key("run", run_id, "reserve")
        entry = self.get_ledger_entry(key)
        if entry is None:
            raise IdempotencyError("Reservation entry not found.")
        settled = validate_settle_entry(entry)
        if settled:
            charged_user = Decimal(str(entry.meta.get("charged_user_sc", "0")))
            charged_pool = Decimal(str(entry.meta.get("charged_pool_sc", "0")))
            refunded_user = Decimal(str(entry.meta.get("refunded_user_sc", "0")))
            refunded_pool = Decimal(str(entry.meta.get("refunded_pool_sc", "0")))
            user_balance = self.get_user_balance(user_id) or UserBalance(user_id=user_id, balance_sc=Decimal("0"))
            return SettlementOutcome(
                entry=entry,
                user_balance=user_balance,
                pool_balance=self.get_pool_balance(),
                charged_user_sc=charged_user,
                charged_pool_sc=charged_pool,
                refunded_user_sc=refunded_user,
                refunded_pool_sc=refunded_pool,
            )

        reserved_user = Decimal(str(entry.meta.get("reserved_user_sc", "0")))
        reserved_pool = Decimal(str(entry.meta.get("reserved_pool_sc", "0")))
        reserved_total = reserved_user + reserved_pool
        extra_user = Decimal("0")
        extra_pool = Decimal("0")
        if actual_sc > reserved_total:
            extra_needed = quantize_sc(actual_sc - reserved_total)
            extra_user = min(user_balance.balance_sc, extra_needed)
            extra_needed = quantize_sc(extra_needed - extra_user)
            extra_pool = min(self.get_pool_balance().spendable_sc, extra_needed)
            if extra_user + extra_pool < actual_sc - reserved_total:
                raise InsufficientFundsError("Insufficient funds to settle actual cost.")
            charge_user = quantize_sc(reserved_user + extra_user)
            charge_pool = quantize_sc(reserved_pool + extra_pool)
            refund_user = Decimal("0")
            refund_pool = Decimal("0")
        else:
            charge_user = min(actual_sc, reserved_user)
            remaining = quantize_sc(actual_sc - charge_user)
            charge_pool = min(remaining, reserved_pool)
            refund_user = quantize_sc(reserved_user - charge_user)
            refund_pool = quantize_sc(reserved_pool - charge_pool)

        user_balance = self.get_user_balance(user_id) or UserBalance(user_id=user_id, balance_sc=Decimal("0"))
        pool_balance = self.get_pool_balance()
        new_user_balance = UserBalance(
            user_id=user_id,
            balance_sc=quantize_sc(user_balance.balance_sc - extra_user + refund_user),
            legacy_credits=user_balance.legacy_credits,
            subsidy_allowance=user_balance.subsidy_allowance,
        )
        new_pool_balance = PoolBalance(
            available_balance_sc=quantize_sc(pool_balance.available_balance_sc - extra_pool + refund_pool),
            locked_buckets=dict(pool_balance.locked_buckets),
            reserve_sc=pool_balance.reserve_sc,
        )

        updated = self.update_ledger_status(
            key,
            LedgerEntryStatus.SETTLED,
            meta={
                "actual_sc": str(actual_sc),
                "charged_user_sc": str(charge_user),
                "charged_pool_sc": str(charge_pool),
                "refunded_user_sc": str(refund_user),
                "refunded_pool_sc": str(refund_pool),
                **(meta or {}),
            },
        )
        self._user_balances[user_id] = new_user_balance
        self._pool_balance = new_pool_balance
        return SettlementOutcome(
            entry=updated,
            user_balance=new_user_balance,
            pool_balance=new_pool_balance,
            charged_user_sc=charge_user,
            charged_pool_sc=charge_pool,
            refunded_user_sc=refund_user,
            refunded_pool_sc=refund_pool,
        )
