# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal

from spectrue_core.monetization.types import MoneySC, PoolBalance, quantize_sc


def deposit(
    pool: PoolBalance,
    amount_sc: MoneySC,
    *,
    lock_ratio: Decimal,
    lock_days: int,
    now: datetime | None = None,
) -> PoolBalance:
    if amount_sc <= 0:
        return pool
    now = now or datetime.utcnow()
    locked_amount = quantize_sc(amount_sc * lock_ratio)
    available_amount = quantize_sc(amount_sc - locked_amount)

    locked_buckets = deepcopy(dict(pool.locked_buckets))
    if locked_amount > 0:
        release_date = (now + timedelta(days=lock_days)).strftime("%Y-%m-%d")
        locked_buckets[release_date] = quantize_sc(
            locked_buckets.get(release_date, Decimal("0")) + locked_amount
        )

    return PoolBalance(
        available_balance_sc=quantize_sc(pool.available_balance_sc + available_amount),
        locked_buckets=locked_buckets,
        reserve_sc=pool.reserve_sc,
    )


def release_matured(pool: PoolBalance, *, as_of: datetime) -> tuple[PoolBalance, MoneySC]:
    released = Decimal("0")
    locked_buckets = deepcopy(dict(pool.locked_buckets))
    for key, amount in list(locked_buckets.items()):
        try:
            bucket_date = datetime.strptime(key, "%Y-%m-%d")
        except ValueError:
            continue
        if bucket_date.date() <= as_of.date():
            released += amount
            del locked_buckets[key]

    if released <= 0:
        return pool, Decimal("0")

    updated = PoolBalance(
        available_balance_sc=quantize_sc(pool.available_balance_sc + released),
        locked_buckets=locked_buckets,
        reserve_sc=pool.reserve_sc,
    )
    return updated, quantize_sc(released)


def deduct(pool: PoolBalance, amount_sc: MoneySC) -> tuple[PoolBalance, bool]:
    if amount_sc <= 0:
        return pool, True
    if pool.available_balance_sc < amount_sc:
        return pool, False
    updated = PoolBalance(
        available_balance_sc=quantize_sc(pool.available_balance_sc - amount_sc),
        locked_buckets=dict(pool.locked_buckets),
        reserve_sc=pool.reserve_sc,
    )
    return updated, True
