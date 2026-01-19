# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations
import logging

from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

try:
    from firebase_admin import firestore
except ImportError:
    firestore = None  # type: ignore

from spectrue_core.monetization.config import MonetizationConfig
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
from spectrue_core.monetization.services.free_pool import deduct as pool_deduct
from spectrue_core.monetization.types import MoneySC, PoolBalance, UserBalance, quantize_sc

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spectrue_core.monetization.types import (
        ChargeResult,
        ChargeSplit,
        DailyBonusState,
        FreePool,
        UserWallet,
        BonusLedgerEntry,  # Added
    )
    from spectrue_core.monetization.ledger import (
        LedgerEntry,
        LedgerEntryType,
    )




def _to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _timestamp_to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_datetime"):
        return value.to_datetime()
    return datetime.utcnow()


class FirestoreBillingStore(BillingStore):
    def __init__(self, db: firestore.Client, *, config: MonetizationConfig | None = None) -> None:
        if firestore is None:
            raise ImportError(
                "FirestoreBillingStore requires 'firebase-admin' to be installed. "
                "Run 'pip install firebase-admin' to use this adapter."
            )
        self._db = db
        self._config = config or MonetizationConfig()
        self._users = self._db.collection(self._config.user_collection)
        self._ledger = self._db.collection(self._config.ledger_collection)
        self._pool_ref = self._db.document(self._config.pool_doc_path)

    def get_user_balance(self, user_id: str) -> UserBalance | None:
        snapshot = self._users.document(user_id).get()
        if not snapshot.exists:
            return None
        return self._user_balance_from_snapshot(snapshot)

    def set_user_balance(self, user_id: str, balance_sc: MoneySC) -> UserBalance:
        user_ref = self._users.document(user_id)
        user_ref.set({self._config.balance_field: str(balance_sc)}, merge=True)
        return UserBalance(user_id=user_id, balance_sc=balance_sc)

    def get_pool_balance(self) -> PoolBalance:
        snapshot = self._pool_ref.get()
        return self._pool_from_snapshot(snapshot)

    def set_pool_balance(self, pool: PoolBalance) -> None:
        self._pool_ref.set(self._pool_to_dict(pool), merge=True)

    def get_ledger_entry(self, idempotency_key: str) -> LedgerEntry | None:
        snapshot = self._ledger.document(idempotency_key).get()
        if not snapshot.exists:
            return None
        return self._entry_from_dict(snapshot.to_dict() or {})

    def write_ledger_entry(self, entry: LedgerEntry) -> LedgerEntry:
        doc_ref = self._ledger.document(entry.idempotency_key)
        snapshot = doc_ref.get()
        if snapshot.exists:
            existing = self._entry_from_dict(snapshot.to_dict() or {})
            if existing == entry:
                return existing
            raise IdempotencyError("Ledger entry already exists for idempotency key.")
        doc_ref.set(self._entry_to_dict(entry))
        return entry

    def update_ledger_status(
        self,
        idempotency_key: str,
        status: LedgerEntryStatus,
        *,
        meta: dict | None = None,
    ) -> LedgerEntry:
        doc_ref = self._ledger.document(idempotency_key)
        snapshot = doc_ref.get()
        if not snapshot.exists:
            raise IdempotencyError("Ledger entry not found for idempotency key.")
        existing = self._entry_from_dict(snapshot.to_dict() or {})
        merged_meta = dict(existing.meta)
        if meta:
            merged_meta.update(meta)
        updated = replace(existing, status=status, meta=merged_meta)
        doc_ref.set(self._entry_to_dict(updated), merge=True)
        return updated

    def count_open_reservations(self, user_id: str) -> int:
        query = (
            self._ledger.where("user_id", "==", user_id)
            .where("entry_type", "==", LedgerEntryType.RESERVE.value)
            .where("status", "==", LedgerEntryStatus.PENDING.value)
        )
        return sum(1 for _ in query.stream())

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
        transaction = self._db.transaction()

        @firestore.transactional
        def _reserve(transaction):  # type: ignore[no-untyped-def]
            ledger_ref = self._ledger.document(key)
            ledger_snapshot = ledger_ref.get(transaction=transaction)
            user_ref = self._users.document(user_id)
            user_snapshot = user_ref.get(transaction=transaction)
            pool_snapshot = self._pool_ref.get(transaction=transaction)

            if ledger_snapshot.exists:
                entry = self._entry_from_dict(ledger_snapshot.to_dict() or {})
                reserved_user = _to_decimal(entry.meta.get("reserved_user_sc", "0"))
                reserved_pool = _to_decimal(entry.meta.get("reserved_pool_sc", "0"))
                user_balance = self._user_balance_from_snapshot(user_snapshot)
                return ReservationOutcome(
                    entry=entry,
                    user_balance=user_balance,
                    pool_balance=self._pool_from_snapshot(pool_snapshot),
                    reserved_user_sc=reserved_user,
                    reserved_pool_sc=reserved_pool,
                )

            open_query = (
                self._ledger.where("user_id", "==", user_id)
                .where("entry_type", "==", LedgerEntryType.RESERVE.value)
                .where("status", "==", LedgerEntryStatus.PENDING.value)
            )
            open_count = sum(1 for _ in open_query.stream(transaction=transaction))
            user_balance = self._user_balance_from_snapshot(user_snapshot)
            pool_balance = self._pool_from_snapshot(pool_snapshot)

            if open_count >= max_parallel_reservations:
                entry = LedgerEntry(
                    idempotency_key=key,
                    entry_type=LedgerEntryType.RESERVE,
                    amount_sc=estimated_sc,
                    status=LedgerEntryStatus.FAILED,
                    user_id=user_id,
                    meta={"reason": "reservation_limit", **(meta or {})},
                )
                ledger_ref.set(self._entry_to_dict(entry))
                return ReservationOutcome(
                    entry=entry,
                    user_balance=user_balance,
                    pool_balance=pool_balance,
                    reserved_user_sc=Decimal("0"),
                    reserved_pool_sc=Decimal("0"),
                )

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
                ledger_ref.set(self._entry_to_dict(entry))
                return ReservationOutcome(
                    entry=entry,
                    user_balance=user_balance,
                    pool_balance=pool_balance,
                    reserved_user_sc=reserved_user,
                    reserved_pool_sc=reserved_pool,
                )

            new_user_balance = quantize_sc(user_balance.balance_sc - reserved_user)
            updated_pool, ok = pool_deduct(pool_balance, reserved_pool)
            if not ok:
                entry = LedgerEntry(
                    idempotency_key=key,
                    entry_type=LedgerEntryType.RESERVE,
                    amount_sc=estimated_sc,
                    status=LedgerEntryStatus.FAILED,
                    user_id=user_id,
                    meta={"reason": "pool_insufficient", **(meta or {})},
                )
                ledger_ref.set(self._entry_to_dict(entry))
                return ReservationOutcome(
                    entry=entry,
                    user_balance=user_balance,
                    pool_balance=pool_balance,
                    reserved_user_sc=Decimal("0"),
                    reserved_pool_sc=Decimal("0"),
                )

            daily_remaining = user_snapshot.get("subsidy_daily_remaining_sc")
            monthly_remaining = user_snapshot.get("subsidy_monthly_remaining_sc")
            update_allowance = daily_remaining is not None or monthly_remaining is not None
            allowance_updates = {}
            if update_allowance and reserved_pool > 0:
                daily_remaining_sc = _to_decimal(daily_remaining)
                monthly_remaining_sc = _to_decimal(monthly_remaining)
                allowance_updates = {
                    "subsidy_daily_remaining_sc": str(
                        quantize_sc(max(Decimal("0"), daily_remaining_sc - reserved_pool))
                    ),
                    "subsidy_monthly_remaining_sc": str(
                        quantize_sc(max(Decimal("0"), monthly_remaining_sc - reserved_pool))
                    ),
                }

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
            ledger_ref.set(self._entry_to_dict(entry))
            transaction.set(
                user_ref,
                {self._config.balance_field: str(new_user_balance), **allowance_updates},
                merge=True,
            )
            transaction.set(self._pool_ref, self._pool_to_dict(updated_pool), merge=True)
            return ReservationOutcome(
                entry=entry,
                user_balance=UserBalance(
                    user_id=user_id,
                    balance_sc=new_user_balance,
                    legacy_credits=user_balance.legacy_credits,
                    subsidy_allowance=user_balance.subsidy_allowance,
                ),
                pool_balance=updated_pool,
                reserved_user_sc=reserved_user,
                reserved_pool_sc=reserved_pool,
            )

        return _reserve(transaction)

    def settle_run(
        self,
        *,
        user_id: str,
        run_id: str,
        actual_sc: MoneySC,
        meta: dict | None = None,
    ) -> SettlementOutcome:
        key = build_idempotency_key("run", run_id, "reserve")
        transaction = self._db.transaction()

        @firestore.transactional
        def _settle(transaction):  # type: ignore[no-untyped-def]
            ledger_ref = self._ledger.document(key)
            ledger_snapshot = ledger_ref.get(transaction=transaction)
            if not ledger_snapshot.exists:
                raise IdempotencyError("Reservation entry not found.")

            entry = self._entry_from_dict(ledger_snapshot.to_dict() or {})
            user_ref = self._users.document(user_id)
            user_snapshot = user_ref.get(transaction=transaction)
            pool_snapshot = self._pool_ref.get(transaction=transaction)
            user_balance = self._user_balance_from_snapshot(user_snapshot)
            pool_balance = self._pool_from_snapshot(pool_snapshot)

            settled = validate_settle_entry(entry)
            if settled:
                charged_user = _to_decimal(entry.meta.get("charged_user_sc", "0"))
                charged_pool = _to_decimal(entry.meta.get("charged_pool_sc", "0"))
                refunded_user = _to_decimal(entry.meta.get("refunded_user_sc", "0"))
                refunded_pool = _to_decimal(entry.meta.get("refunded_pool_sc", "0"))
                return SettlementOutcome(
                    entry=entry,
                    user_balance=user_balance,
                    pool_balance=pool_balance,
                    charged_user_sc=charged_user,
                    charged_pool_sc=charged_pool,
                    refunded_user_sc=refunded_user,
                    refunded_pool_sc=refunded_pool,
                )

            reserved_user = _to_decimal(entry.meta.get("reserved_user_sc", "0"))
            reserved_pool = _to_decimal(entry.meta.get("reserved_pool_sc", "0"))
            reserved_total = reserved_user + reserved_pool
            extra_user = Decimal("0")
            extra_pool = Decimal("0")
            if actual_sc > reserved_total:
                extra_needed = quantize_sc(actual_sc - reserved_total)
                extra_user = min(user_balance.balance_sc, extra_needed)
                extra_needed = quantize_sc(extra_needed - extra_user)
                extra_pool = min(pool_balance.spendable_sc, extra_needed)
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

            daily_remaining = user_snapshot.get("subsidy_daily_remaining_sc")
            monthly_remaining = user_snapshot.get("subsidy_monthly_remaining_sc")
            update_allowance = daily_remaining is not None or monthly_remaining is not None
            allowance_updates = {}
            if update_allowance and (refund_pool > 0 or extra_pool > 0):
                daily_remaining_sc = _to_decimal(daily_remaining)
                monthly_remaining_sc = _to_decimal(monthly_remaining)
                allowance_updates = {
                    "subsidy_daily_remaining_sc": str(
                        quantize_sc(max(Decimal("0"), daily_remaining_sc - extra_pool + refund_pool))
                    ),
                    "subsidy_monthly_remaining_sc": str(
                        quantize_sc(max(Decimal("0"), monthly_remaining_sc - extra_pool + refund_pool))
                    ),
                }

            updated_user_balance = quantize_sc(user_balance.balance_sc - extra_user + refund_user)
            updated_pool = PoolBalance(
                available_balance_sc=quantize_sc(pool_balance.available_balance_sc - extra_pool + refund_pool),
                locked_buckets=dict(pool_balance.locked_buckets),
                reserve_sc=pool_balance.reserve_sc,
            )

            updated_entry = LedgerEntry(
                idempotency_key=entry.idempotency_key,
                entry_type=entry.entry_type,
                amount_sc=entry.amount_sc,
                status=LedgerEntryStatus.SETTLED,
                created_at=entry.created_at,
                event_id=entry.event_id,
                user_id=entry.user_id,
                meta={
                    **entry.meta,
                    "actual_sc": str(actual_sc),
                    "charged_user_sc": str(charge_user),
                    "charged_pool_sc": str(charge_pool),
                    "refunded_user_sc": str(refund_user),
                    "refunded_pool_sc": str(refund_pool),
                    **(meta or {}),
                },
            )
            ledger_ref.set(self._entry_to_dict(updated_entry), merge=True)
            transaction.set(
                user_ref,
                {self._config.balance_field: str(updated_user_balance), **allowance_updates},
                merge=True,
            )
            transaction.set(self._pool_ref, self._pool_to_dict(updated_pool), merge=True)

            return SettlementOutcome(
                entry=updated_entry,
                user_balance=UserBalance(
                    user_id=user_id,
                    balance_sc=updated_user_balance,
                    legacy_credits=user_balance.legacy_credits,
                    subsidy_allowance=user_balance.subsidy_allowance,
                ),
                pool_balance=updated_pool,
                charged_user_sc=charge_user,
                charged_pool_sc=charge_pool,
                refunded_user_sc=refund_user,
                refunded_pool_sc=refund_pool,
            )

        return _settle(transaction)

    def _user_balance_from_snapshot(self, snapshot) -> UserBalance:
        if not snapshot.exists:
            return UserBalance(user_id=snapshot.id, balance_sc=Decimal("0"))
        data = snapshot.to_dict() or {}
        raw_balance = data.get(self._config.balance_field)
        raw_legacy = data.get(self._config.legacy_balance_field)
        if raw_balance is None and raw_legacy is not None:
            balance_sc = _to_decimal(raw_legacy)
            legacy_sc = _to_decimal(raw_legacy)
        else:
            balance_sc = _to_decimal(raw_balance)
            legacy_sc = _to_decimal(raw_legacy) if raw_legacy is not None else None
        return UserBalance(user_id=snapshot.id, balance_sc=balance_sc, legacy_credits=legacy_sc)

    def _pool_from_snapshot(self, snapshot) -> PoolBalance:
        data = snapshot.to_dict() if snapshot.exists else {}
        available = _to_decimal(data.get("available_balance_sc"))
        locked_buckets = {
            key: _to_decimal(value) for key, value in (data.get("locked_buckets") or {}).items()
        }
        reserve = data.get("reserve_sc")
        reserve_sc = _to_decimal(reserve, self._config.pool_reserve_sc)
        return PoolBalance(
            available_balance_sc=available,
            locked_buckets=locked_buckets,
            reserve_sc=reserve_sc,
        )

    def _pool_to_dict(self, pool: PoolBalance) -> dict[str, Any]:
        return {
            "available_balance_sc": str(pool.available_balance_sc),
            "locked_buckets": {k: str(v) for k, v in pool.locked_buckets.items()},
            "reserve_sc": str(pool.reserve_sc),
            "updated_at": firestore.SERVER_TIMESTAMP,
        }

    def _entry_from_dict(self, data: dict[str, Any]) -> LedgerEntry:
        created_at = _timestamp_to_datetime(data.get("created_at"))
        return LedgerEntry(
            idempotency_key=data.get("idempotency_key", ""),
            entry_type=LedgerEntryType(data.get("entry_type", LedgerEntryType.RESERVE.value)),
            amount_sc=_to_decimal(data.get("amount_sc")),
            status=LedgerEntryStatus(data.get("status", LedgerEntryStatus.PENDING.value)),
            created_at=created_at,
            event_id=data.get("event_id"),
            user_id=data.get("user_id"),
            meta=data.get("meta") or {},
        )

    def _entry_to_dict(self, entry: LedgerEntry) -> dict[str, Any]:
        return {
            "idempotency_key": entry.idempotency_key,
            "entry_type": entry.entry_type.value,
            "amount_sc": str(entry.amount_sc),
            "status": entry.status.value,
            "created_at": entry.created_at,
            "event_id": entry.event_id,
            "user_id": entry.user_id,
            "meta": entry.meta,
        }

    # =========================================================================
    # Methods: UserWallet, FreePool, DailyBonus, Charging
    # =========================================================================

    def read_user_wallet(self, uid: str) -> "UserWallet":
        """Read user wallet with available_sc field."""
        from spectrue_core.monetization.types import UserWallet, MoneySC as MoneySCClass
        from datetime import date

        snapshot = self._users.document(uid).get()
        if not snapshot.exists:
            return UserWallet(
                uid=uid,
                balance_sc=MoneySCClass.zero(),
                available_sc=MoneySCClass.zero(),
            )
        data = snapshot.to_dict() or {}
        balance = _to_decimal(data.get(self._config.balance_field, "0"))
        available = _to_decimal(data.get(self._config.available_field, "0"))
        last_seen_at = data.get("last_seen_at")
        if last_seen_at and hasattr(last_seen_at, "to_datetime"):
            last_seen_at = last_seen_at.to_datetime()

        last_daily = data.get("last_daily_bonus_date")
        if isinstance(last_daily, str):
            last_daily = date.fromisoformat(last_daily)
        elif last_daily and hasattr(last_daily, "date"):
            last_daily = last_daily.date()
        else:
            last_daily = None

        last_share = data.get("last_share_bonus_date")
        if isinstance(last_share, str):
            last_share = date.fromisoformat(last_share)
        elif last_share and hasattr(last_share, "date"):
            last_share = last_share.date()
        else:
            last_share = None

        return UserWallet(
            uid=uid,
            balance_sc=MoneySCClass(balance),
            available_sc=MoneySCClass(available),
            last_seen_at=last_seen_at,
            last_daily_bonus_date=last_daily,
            last_share_bonus_date=last_share,
        )

    def read_pool(self) -> "FreePool":
        """Read free pool with locked_buckets list."""
        from spectrue_core.monetization.types import FreePool, LockedBucket, MoneySC as MoneySCClass

        snapshot = self._pool_ref.get()
        if not snapshot.exists:
            # Auto-bootstrap pool doc (atomic). This prevents "phantom credits" and
            # ensures the pool is visible in Firestore emulator immediately.
            tx = self._db.transaction()

            @firestore.transactional
            def _bootstrap_pool(transaction: firestore.Transaction) -> dict:  # type: ignore[no-untyped-def]
                snap = self._pool_ref.get(transaction=transaction)
                if snap.exists:
                    return snap.to_dict() or {}
                bootstrap = {
                    "available_balance_sc": self._config.initial_pool_available_sc.to_str(),
                    "locked_buckets": [],
                    "reserve_sc": self._config.pool_reserve_sc.to_str(),
                    "source": self._config.pool_bootstrap_source,
                    "created_at": firestore.SERVER_TIMESTAMP,
                    "updated_at": firestore.SERVER_TIMESTAMP,
                }
                # Use create() so we never overwrite an existing doc if a race happens.
                transaction.create(self._pool_ref, bootstrap)
                # Firestore forbids read-after-write in the same transaction.
                # Return the bootstrap dict as the source of truth for this call.
                return {
                    **bootstrap,
                    # Normalize server timestamp placeholders for downstream parsing:
                    # parser already defaults missing timestamps safely.
                    "created_at": None,
                    "updated_at": None,
                }

            try:
                data = _bootstrap_pool(tx)
                logger.info(
                    "Bootstrapped free pool doc at %s with available_balance_sc=%s",
                    self._config.pool_doc_path,
                    self._config.initial_pool_available_sc.to_str(),
                )
            except Exception as e:
                logger.exception("Failed to bootstrap free pool doc: %s", e)
                return FreePool(available_balance_sc=MoneySCClass.zero())
        else:
            data = snapshot.to_dict() or {}
        available = _to_decimal(data.get("available_balance_sc", "0"))
        # Support both new clean key and temporary V3 key if migration happened
        raw_buckets = data.get("locked_buckets") or data.get("locked_buckets_v3") or []
        locked_buckets = []
        
        if isinstance(raw_buckets, dict):
             # Handle V2/PoolBalance format (dict[date_str, amount_str])
             from datetime import date
             for date_key, amount_val in raw_buckets.items():
                 try:
                     unlock_dt = date.fromisoformat(date_key)
                     amt = MoneySCClass.from_str(str(amount_val))
                     locked_buckets.append(LockedBucket(amount_sc=amt, unlock_at=unlock_dt))
                 except Exception as e:
                     logger.warning(f"Failed to parse locked bucket {date_key}: {e}")
        elif isinstance(raw_buckets, list):
            for b in raw_buckets:
                if isinstance(b, dict):
                    locked_buckets.append(LockedBucket.from_dict(b))
        updated_at = data.get("updated_at")
        if updated_at and hasattr(updated_at, "to_datetime"):
            updated_at = updated_at.to_datetime()

        reserve_sc = _to_decimal(data.get("reserve_sc"), self._config.pool_reserve_sc.value)

        return FreePool(
            available_balance_sc=MoneySCClass(available),
            locked_buckets=locked_buckets,
            reserve_sc=MoneySCClass(reserve_sc),
            updated_at=updated_at,
        )

    def read_daily_bonus_state(self) -> "DailyBonusState":
        """Read daily bonus job state."""
        from spectrue_core.monetization.types import DailyBonusState, MoneySC as MoneySCClass

        state_ref = self._db.document(self._config.bonus_state_doc_path)
        snapshot = state_ref.get()
        if not snapshot.exists:
            return DailyBonusState(ema_budget_sc=MoneySCClass.zero())
        data = snapshot.to_dict() or {}
        return DailyBonusState.from_dict(data)

    def list_active_users(self, today: "date", window_days: int) -> list[str]:
        """List user IDs that are active (seen within window_days)."""
        from datetime import timedelta
        
        cutoff = datetime.combine(today - timedelta(days=window_days), datetime.min.time())
        query = self._users.where("last_seen_at", ">=", cutoff)
        return [doc.id for doc in query.stream()]

    def read_last_daily_b(self) -> "MoneySC":
        """Read last computed daily bonus per user."""

        state = self.read_daily_bonus_state()
        return state.last_b_sc

    def apply_daily_bonus_batch(
        self,
        user_ids: list[str],
        bonus_per_user: "MoneySC",
        today: "date",
        new_pool: "FreePool",
        new_state: "DailyBonusState",
    ) -> int:
        """Apply daily bonus to a batch of users atomically."""

        if not user_ids:
            # Still persist pool/state updates
            self._pool_ref.set(self._serialize_free_pool(new_pool), merge=True)
            state_ref = self._db.document(self._config.bonus_state_doc_path)
            state_ref.set(new_state.to_dict(), merge=True)
            return 0

        awarded = 0
        batch = self._db.batch()

        for uid in user_ids:
            user_ref = self._users.document(uid)
            snapshot = user_ref.get()
            data = snapshot.to_dict() or {} if snapshot.exists else {}

            # Check if already awarded today
            last_daily = data.get("last_daily_bonus_date")
            if isinstance(last_daily, str):
                from datetime import date as date_type
                last_daily = date_type.fromisoformat(last_daily)
            elif last_daily and hasattr(last_daily, "date"):
                last_daily = last_daily.date()

            if last_daily == today:
                continue  # Already awarded

            current_available = _to_decimal(data.get(self._config.available_field, "0"))
            new_available = current_available + bonus_per_user.value

            batch.set(
                user_ref,
                {
                    self._config.available_field: str(new_available),
                    "last_daily_bonus_date": today.isoformat(),
                },
                merge=True,
            )
            awarded += 1

        # Update pool and state
        batch.set(self._pool_ref, self._serialize_free_pool(new_pool), merge=True)
        state_ref = self._db.document(self._config.bonus_state_doc_path)
        batch.set(state_ref, new_state.to_dict(), merge=True)

        batch.commit()
        return awarded

    def apply_share_bonus(self, uid: str, bonus: "MoneySC", today: "date") -> bool:
        """Apply share bonus to user's available_sc (1/day limit)."""
        transaction = self._db.transaction()

        @firestore.transactional
        def _apply(transaction):  # type: ignore[no-untyped-def]
            user_ref = self._users.document(uid)
            snapshot = user_ref.get(transaction=transaction)
            data = snapshot.to_dict() or {} if snapshot.exists else {}

            last_share = data.get("last_share_bonus_date")
            if isinstance(last_share, str):
                from datetime import date as date_type
                last_share = date_type.fromisoformat(last_share)
            elif last_share and hasattr(last_share, "date"):
                last_share = last_share.date()

            if last_share == today:
                return False  # Already awarded

            current_available = _to_decimal(data.get(self._config.available_field, "0"))
            new_available = current_available + bonus.value

            transaction.set(
                user_ref,
                {
                    self._config.available_field: str(new_available),
                    "last_share_bonus_date": today.isoformat(),
                },
                merge=True,
            )
            return True

        return _apply(transaction)

    # Charge is implemented via apply_charge() below. Keep a single code path.
    def apply_charge(
        self,
        uid: str,
        run_id: str,
        split: "ChargeSplit",
        idempotency_key: str,
    ) -> "ChargeResult":
        """Apply charge with split (available_sc first, then balance_sc)."""
        from spectrue_core.monetization.types import (
            ChargeResult, 
            ChargeSplit, 
            MoneySC as MoneySCClass,
            FreePool,
            LockedBucket
        )

        transaction = self._db.transaction()

        @firestore.transactional
        def _charge(transaction):  # type: ignore[no-untyped-def]
            # Check idempotency
            ledger_ref = self._ledger.document(idempotency_key)
            ledger_snap = ledger_ref.get(transaction=transaction)
            if ledger_snap.exists:
                # Return cached result
                entry_data = ledger_snap.to_dict() or {}
                meta = entry_data.get("meta", {})
                return ChargeResult(
                    ok=entry_data.get("status") == LedgerEntryStatus.SETTLED.value,
                    split=ChargeSplit(
                        take_available=MoneySCClass.from_str(meta.get("take_available_sc", "0")),
                        take_balance=MoneySCClass.from_str(meta.get("take_balance_sc", "0")),
                    ),
                    new_balance_sc=MoneySCClass.from_str(meta.get("new_balance_sc", "0")),
                    new_available_sc=MoneySCClass.from_str(meta.get("new_available_sc", "0")),
                    reason=meta.get("reason"),
                )

            user_ref = self._users.document(uid)
            snapshot = user_ref.get(transaction=transaction)
            data = snapshot.to_dict() or {} if snapshot.exists else {}

            current_balance = _to_decimal(data.get(self._config.balance_field, "0"))
            current_available = _to_decimal(data.get(self._config.available_field, "0"))

            new_available = quantize_sc(current_available - split.take_available.value)
            new_balance = quantize_sc(current_balance - split.take_balance.value)

            # Lazy Spend Logic: Check Pool Liquidity *BEFORE* any writes
            updated_pool_final = None
            if split.take_available.value > 0:
                # 1. Check Pool Liquidity (Real Cash Check)
                pool_snapshot_latest = self._pool_ref.get(transaction=transaction)
                if not pool_snapshot_latest.exists:
                     raise InsufficientFundsError("Pool missing during spend.")
                
                # Custom inline parsing to FreePool (Modern Type) to avoid Legacy PoolBalance issues
                # and ensure MoneySC wrapping.
                p_data = pool_snapshot_latest.to_dict() or {}
                p_avail = MoneySCClass(_to_decimal(p_data.get("available_balance_sc", "0")))
                p_reserve = MoneySCClass(_to_decimal(p_data.get("reserve_sc"), self._config.pool_reserve_sc.value))
                
                # Parse buckets (New schema is list, fallback to empty)
                raw_buckets = p_data.get("locked_buckets") or []
                p_buckets = []
                if isinstance(raw_buckets, list):
                     for b in raw_buckets:
                         if isinstance(b, dict):
                             p_buckets.append(LockedBucket.from_dict(b))
                
                pool_latest = FreePool(
                    available_balance_sc=p_avail,
                    locked_buckets=p_buckets,
                    reserve_sc=p_reserve
                )
                
                spendable_cash = pool_latest.spendable_sc
                
                if spendable_cash < split.take_available:
                     raise InsufficientFundsError(f"Pool exhausted. Available: {pool_latest.available_balance_sc}. Needed: {split.take_available}")

                # 2. Prepare Pool Deduction (Real Spend)
                new_pool_avail = pool_latest.available_balance_sc - split.take_available
                updated_pool_final = FreePool(
                    available_balance_sc=new_pool_avail,
                    locked_buckets=pool_latest.locked_buckets,
                    reserve_sc=pool_latest.reserve_sc
                )

            # --- WRITE PHASE START ---

            # Ledger entry
            entry = LedgerEntry(
                idempotency_key=idempotency_key,
                entry_type=LedgerEntryType.CHARGE,
                amount_sc=split.total.value,
                status=LedgerEntryStatus.SETTLED,
                user_id=uid,
                meta={
                    "run_id": run_id,
                    "take_available_sc": split.take_available.to_str(),
                    "take_balance_sc": split.take_balance.to_str(),
                    "new_available_sc": str(new_available),
                    "new_balance_sc": str(new_balance),
                },
            )
            transaction.set(ledger_ref, self._entry_to_dict(entry))
            transaction.set(
                user_ref,
                {
                    self._config.available_field: str(new_available),
                    self._config.balance_field: str(new_balance),
                },
                merge=True,
            )

            # Pool Update & Bonus Ledger (if applicable)
            if updated_pool_final:
                transaction.set(self._pool_ref, self._serialize_free_pool(updated_pool_final), merge=True)

                # Bonus Ledger (Record the spend event)
                from spectrue_core.monetization.types import BonusLedgerEntry
                from datetime import datetime, timezone
                bonus_entry = BonusLedgerEntry(
                    uid=uid,
                    event_type="spend",
                    amount_sc=split.take_available * -1,
                    date=datetime.now(timezone.utc),
                    related_run_id=run_id,
                    balance_after=MoneySCClass(new_available),
                )
                self.add_bonus_ledger_entry(bonus_entry, transaction=transaction)

            return ChargeResult(
                ok=True,
                split=split,
                new_balance_sc=MoneySCClass(new_balance),
                new_available_sc=MoneySCClass(new_available),
            )

        return _charge(transaction)

    def check_idempotency(self, idempotency_key: str) -> "ChargeResult | None":
        """Check if a charge was already applied."""
        from spectrue_core.monetization.types import ChargeResult, ChargeSplit, MoneySC as MoneySCClass

        snapshot = self._ledger.document(idempotency_key).get()
        if not snapshot.exists:
            return None
        entry_data = snapshot.to_dict() or {}
        meta = entry_data.get("meta", {})
        return ChargeResult(
            ok=entry_data.get("status") == LedgerEntryStatus.SETTLED.value,
            split=ChargeSplit(
                take_available=MoneySCClass.from_str(meta.get("take_available_sc", "0")),
                take_balance=MoneySCClass.from_str(meta.get("take_balance_sc", "0")),
            ),
            new_balance_sc=MoneySCClass.from_str(meta.get("new_balance_sc", "0")),
            new_available_sc=MoneySCClass.from_str(meta.get("new_available_sc", "0")),
            reason=meta.get("reason"),
        )

    def _serialize_free_pool(self, pool: "FreePool") -> dict[str, Any]:
        """Convert FreePool to Firestore dict."""
        return {
            "available_balance_sc": pool.available_balance_sc.to_str(),
            "locked_buckets": [b.to_dict() for b in pool.locked_buckets],
            "updated_at": firestore.SERVER_TIMESTAMP,
        }

    def add_bonus_ledger_entry(
        self,
        entry: "BonusLedgerEntry",
        transaction: Any | None = None,
        batch: Any | None = None,
    ) -> None:
        """Add a bonus ledger entry to Firestore."""
        # Use root collection "bonus_ledger"
        ref = self._db.collection("bonus_ledger").document()
        data = entry.to_dict()
        data["created_at"] = firestore.SERVER_TIMESTAMP

        if transaction:
            transaction.set(ref, data)
        elif batch:
            batch.set(ref, data)
        else:
            ref.set(data)
