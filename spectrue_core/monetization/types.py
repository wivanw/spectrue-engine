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
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from typing import Mapping

MoneySC = Decimal
SC_PLACES = 6
SC_QUANT = Decimal("0.000001")


def to_money_sc(value: MoneySC | float | int | str) -> MoneySC:
    return Decimal(str(value))


def quantize_sc(value: MoneySC, rounding=ROUND_HALF_UP) -> MoneySC:
    return value.quantize(SC_QUANT, rounding=rounding)


def ceil_sc(value: MoneySC) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def sc_to_str(value: MoneySC) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))
    return format(normalized, "f")


@dataclass(frozen=True, slots=True)
class EligibilityAllowance:
    eligible: bool
    daily_remaining_sc: MoneySC
    monthly_remaining_sc: MoneySC
    reason: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "eligible": self.eligible,
            "daily_remaining_sc": sc_to_str(self.daily_remaining_sc),
            "monthly_remaining_sc": sc_to_str(self.monthly_remaining_sc),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class UserBalance:
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
    balance_sc: MoneySC
    legacy_credits: MoneySC | None = None
    subsidy_allowance: EligibilityAllowance | None = None

    def to_dict(self) -> dict[str, str | dict | None]:
        payload: dict[str, str | dict | None] = {
            "balance_sc": sc_to_str(self.balance_sc),
        }
        if self.legacy_credits is not None:
            payload["legacy_credits"] = sc_to_str(self.legacy_credits)
        if self.subsidy_allowance is not None:
            payload["subsidy"] = self.subsidy_allowance.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class PoolBalance:
    available_balance_sc: MoneySC
    locked_buckets: Mapping[str, MoneySC] = field(default_factory=dict)
    reserve_sc: MoneySC = Decimal("0")

    @property
    def locked_sc(self) -> MoneySC:
        return sum(self.locked_buckets.values(), Decimal("0"))

    @property
    def total_sc(self) -> MoneySC:
        return self.available_balance_sc + self.locked_sc

    @property
    def spendable_sc(self) -> MoneySC:
        spendable = self.available_balance_sc - self.reserve_sc
        return spendable if spendable > 0 else Decimal("0")

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
    total_sc: MoneySC
    locked_sc: MoneySC
    available_sc: MoneySC
    reserve_sc: MoneySC
    spendable_sc: MoneySC

    def to_dict(self) -> dict[str, str]:
        return {
            "total_sc": sc_to_str(self.total_sc),
            "locked_sc": sc_to_str(self.locked_sc),
            "available_sc": sc_to_str(self.available_sc),
            "reserve_sc": sc_to_str(self.reserve_sc),
            "spendable_sc": sc_to_str(self.spendable_sc),
        }
