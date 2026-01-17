# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from typing import List, Mapping, Optional

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SC_PLACES = 6
SC_QUANT = Decimal("0.000001")


# -----------------------------------------------------------------------------
# Money Helpers (Decimal-based, no floats)
# -----------------------------------------------------------------------------

def quantize_sc(value: Decimal, rounding=ROUND_HALF_UP) -> Decimal:
    """Quantize a Decimal to SC precision."""
    return value.quantize(SC_QUANT, rounding=rounding)


def ceil_sc(value: Decimal) -> int:
    """Ceiling integer of a Decimal."""
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def sc_to_str(value: Decimal) -> str:
    """Convert Decimal to string, stripping unnecessary zeros."""
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f")


def to_money_sc(value: Decimal | float | int | str) -> Decimal:
    """Convert any numeric to Decimal."""
    return Decimal(str(value))


# -----------------------------------------------------------------------------
# MoneySC Dataclass (frozen, safe arithmetic)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MoneySC:
    """Immutable money value with automatic quantization."""
    value: Decimal

    def __post_init__(self):
        # Quantize on creation
        object.__setattr__(self, "value", quantize_sc(self.value))

    def __add__(self, other: "MoneySC") -> "MoneySC":
        return MoneySC(self.value + other.value)

    def __sub__(self, other: "MoneySC") -> "MoneySC":
        return MoneySC(self.value - other.value)

    def __mul__(self, factor: Decimal | int | float) -> "MoneySC":
        return MoneySC(self.value * Decimal(str(factor)))

    def __truediv__(self, divisor: Decimal | int | float) -> "MoneySC":
        if Decimal(str(divisor)) == 0:
            raise ZeroDivisionError("Cannot divide MoneySC by zero")
        return MoneySC(self.value / Decimal(str(divisor)))

    def __lt__(self, other: "MoneySC") -> bool:
        return self.value < other.value

    def __le__(self, other: "MoneySC") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "MoneySC") -> bool:
        return self.value > other.value

    def __ge__(self, other: "MoneySC") -> bool:
        return self.value >= other.value

    def min(self, other: "MoneySC") -> "MoneySC":
        return MoneySC(min(self.value, other.value))

    def max(self, other: "MoneySC") -> "MoneySC":
        return MoneySC(max(self.value, other.value))

    def max0(self) -> "MoneySC":
        """Return max(0, self)."""
        return MoneySC(max(Decimal("0"), self.value))

    def to_str(self) -> str:
        return sc_to_str(self.value)

    @classmethod
    def zero(cls) -> "MoneySC":
        return cls(Decimal("0"))

    @classmethod
    def from_str(cls, s: str) -> "MoneySC":
        return cls(Decimal(s))


# Alias for backward compatibility
MoneyDecimal = Decimal


# -----------------------------------------------------------------------------
# Eligibility Allowance (subsidy eligibility)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EligibilityAllowance:
    eligible: bool
    daily_remaining_sc: MoneySC
    monthly_remaining_sc: MoneySC
    reason: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "eligible": self.eligible,
            "daily_remaining_sc": self.daily_remaining_sc.to_str(),
            "monthly_remaining_sc": self.monthly_remaining_sc.to_str(),
            "reason": self.reason,
        }


# -----------------------------------------------------------------------------
# User Wallet (v3: balance_sc + available_sc)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class UserWallet:
    """User wallet with paid balance and bonus credits."""
    uid: str
    balance_sc: MoneySC          # Paid / owned funds (NOT bonuses)
    available_sc: MoneySC        # Bonus credits available to spend
    last_seen_at: Optional[datetime] = None
    last_daily_bonus_date: Optional[date] = None
    last_share_bonus_date: Optional[date] = None

    def total_sc(self) -> MoneySC:
        """Total spendable credits (available + balance)."""
        return self.available_sc + self.balance_sc

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "balance_sc": self.balance_sc.to_str(),
            "available_sc": self.available_sc.to_str(),
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "last_daily_bonus_date": self.last_daily_bonus_date.isoformat() if self.last_daily_bonus_date else None,
            "last_share_bonus_date": self.last_share_bonus_date.isoformat() if self.last_share_bonus_date else None,
        }


# -----------------------------------------------------------------------------
# Charge Split & Result (v3 charging)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChargeSplit:
    """How a charge is split between available and balance."""
    take_available: MoneySC
    take_balance: MoneySC

    @property
    def total(self) -> MoneySC:
        return self.take_available + self.take_balance


@dataclass(frozen=True, slots=True)
class ChargeResult:
    """Result of a charge operation."""
    ok: bool
    split: ChargeSplit
    new_balance_sc: MoneySC
    new_available_sc: MoneySC
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "take_available": self.split.take_available.to_str(),
            "take_balance": self.split.take_balance.to_str(),
            "total_charged": self.split.total.to_str(),
            "new_balance_sc": self.new_balance_sc.to_str(),
            "new_available_sc": self.new_available_sc.to_str(),
            "reason": self.reason,
        }


# -----------------------------------------------------------------------------
# Locked Bucket & Free Pool V3
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LockedBucket:
    """A bucket of funds locked until a maturity date."""
    amount_sc: MoneySC
    unlock_at: date  # 90-day maturity (or as configured)

    def to_dict(self) -> dict:
        return {
            "amount_sc": self.amount_sc.to_str(),
            "unlock_at": self.unlock_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LockedBucket":
        return cls(
            amount_sc=MoneySC.from_str(d["amount_sc"]),
            unlock_at=date.fromisoformat(d["unlock_at"]),
        )


@dataclass(frozen=True, slots=True)
class FreePoolV3:
    """Free subsidy pool with available balance and locked buckets."""
    available_balance_sc: MoneySC
    locked_buckets: List[LockedBucket] = field(default_factory=list)
    updated_at: Optional[datetime] = None

    def locked_total(self) -> MoneySC:
        """Sum of all locked bucket amounts."""
        total = Decimal("0")
        for b in self.locked_buckets:
            total += b.amount_sc.value
        return MoneySC(total)

    def total(self) -> MoneySC:
        """Total pool = available + locked."""
        return self.available_balance_sc + self.locked_total()

    def to_dict(self) -> dict:
        return {
            "available_balance_sc": self.available_balance_sc.to_str(),
            "locked_buckets": [b.to_dict() for b in self.locked_buckets],
            "locked_total_sc": self.locked_total().to_str(),
            "total_sc": self.total().to_str(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# -----------------------------------------------------------------------------
# Daily Bonus State (EMA budget tracking)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DailyBonusState:
    """State for daily bonus EMA budget calculation."""
    ema_budget_sc: MoneySC               # EMA(B) smoothed budget
    last_run_date: Optional[date] = None
    smoothing_alpha: Decimal = field(default=Decimal("0.3"))
    active_user_count: int = 0
    last_b_sc: MoneySC = field(default_factory=MoneySC.zero)

    def to_dict(self) -> dict:
        return {
            "ema_budget_sc": self.ema_budget_sc.to_str(),
            "last_run_date": self.last_run_date.isoformat() if self.last_run_date else None,
            "smoothing_alpha": str(self.smoothing_alpha),
            "active_user_count": self.active_user_count,
            "last_b_sc": self.last_b_sc.to_str(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DailyBonusState":
        return cls(
            ema_budget_sc=MoneySC.from_str(d.get("ema_budget_sc", "0")),
            last_run_date=date.fromisoformat(d["last_run_date"]) if d.get("last_run_date") else None,
            smoothing_alpha=Decimal(d.get("smoothing_alpha", "0.3")),
            active_user_count=int(d.get("active_user_count", 0)),
            last_b_sc=MoneySC.from_str(d.get("last_b_sc", "0")),
        )


# -----------------------------------------------------------------------------
# Bonus Ledger
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BonusLedgerEntry:
    """Entry in the bonus credits ledger."""
    uid: str
    event_type: str  # daily_bonus, share_bonus, spend
    amount_sc: MoneySC
    date: datetime
    related_run_id: Optional[str] = None
    balance_after: Optional[MoneySC] = None  # available_sc after this op

    def to_dict(self) -> dict:
        d = {
            "uid": self.uid,
            "event_type": self.event_type,
            "amount_sc": self.amount_sc.to_str(),
            "date": self.date.isoformat(),
        }
        if self.related_run_id:
            d["related_run_id"] = self.related_run_id
        if self.balance_after:
            d["balance_after"] = self.balance_after.to_str()
        return d


# -----------------------------------------------------------------------------
# Legacy Types (backward compatibility)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class UserBalance:
    """Legacy user balance type (kept for compatibility)."""
    user_id: str
    balance_sc: MoneySC
    legacy_credits: MoneySC | None = None
    subsidy_allowance: EligibilityAllowance | None = None

    def to_stats(self) -> "UserBalanceStats":
        return UserBalanceStats(
            balance_sc=self.balance_sc,
            legacy_credits=self.legacy_credits,
            subsidy_allowance=self.subsidy_allowance,
        )


@dataclass(frozen=True, slots=True)
class UserBalanceStats:
    """Legacy user balance stats (kept for compatibility)."""
    balance_sc: MoneySC
    legacy_credits: MoneySC | None = None
    subsidy_allowance: EligibilityAllowance | None = None

    def to_dict(self) -> dict[str, str | dict | None]:
        payload: dict[str, str | dict | None] = {
            "balance_sc": self.balance_sc.to_str(),
        }
        if self.legacy_credits is not None:
            payload["legacy_credits"] = self.legacy_credits.to_str()
        if self.subsidy_allowance is not None:
            payload["subsidy"] = self.subsidy_allowance.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class PoolBalance:
    """Legacy pool balance (kept for compatibility)."""
    available_balance_sc: MoneySC
    locked_buckets: Mapping[str, MoneySC] = field(default_factory=dict)
    reserve_sc: MoneySC = field(default_factory=MoneySC.zero)

    @property
    def locked_sc(self) -> MoneySC:
        total = Decimal("0")
        for v in self.locked_buckets.values():
            total += v.value
        return MoneySC(total)

    @property
    def total_sc(self) -> MoneySC:
        return self.available_balance_sc + self.locked_sc

    @property
    def spendable_sc(self) -> MoneySC:
        spendable = self.available_balance_sc - self.reserve_sc
        return spendable.max0()

    def to_stats(self) -> "PoolStats":
        return PoolStats(
            total_sc=self.total_sc,
            locked_sc=self.locked_sc,
            available_sc=self.available_balance_sc,
            reserve_sc=self.reserve_sc,
            spendable_sc=self.spendable_sc,
        )


@dataclass(frozen=True, slots=True)
class PoolStats:
    """Legacy pool stats (kept for compatibility)."""
    total_sc: MoneySC
    locked_sc: MoneySC
    available_sc: MoneySC
    reserve_sc: MoneySC
    spendable_sc: MoneySC

    def to_dict(self) -> dict[str, str]:
        return {
            "total_sc": self.total_sc.to_str(),
            "locked_sc": self.locked_sc.to_str(),
            "available_sc": self.available_sc.to_str(),
            "reserve_sc": self.reserve_sc.to_str(),
            "spendable_sc": self.spendable_sc.to_str(),
        }
