# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Share Bonus Service for Monetization V3.

This service handles the share bonus:
- User shares content and receives 20% of daily b
- Limited to 1 share bonus per day per user
- Bonus goes to available_sc
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Protocol

from spectrue_core.monetization.types import MoneySC, UserWallet

if TYPE_CHECKING:
    from spectrue_core.monetization.config import MonetizationConfig


class ShareBonusStore(Protocol):
    """Protocol for share bonus storage operations."""

    def read_user_wallet(self, uid: str) -> UserWallet:
        """Read user wallet."""
        ...

    def apply_share_bonus(
        self,
        uid: str,
        bonus: MoneySC,
        today: date,
    ) -> bool:
        """
        Apply share bonus to user's available_sc.
        Returns True if bonus was applied, False if user already received share bonus today.
        """
        ...

    def read_last_daily_b(self) -> MoneySC:
        """Read the last computed daily bonus per user (b) from bonus state."""
        ...


class ShareBonusService:
    """Service for applying share bonuses."""

    def __init__(self, store: ShareBonusStore, cfg: "MonetizationConfig"):
        self.store = store
        self.cfg = cfg

    def compute_share_bonus(self, base_b: MoneySC) -> MoneySC:
        """Compute share bonus as a ratio of the daily bonus."""
        return MoneySC(base_b.value * self.cfg.share_ratio)

    def apply_share_bonus(
        self,
        uid: str,
        today: date | None = None,
        base_b: MoneySC | None = None,
    ) -> dict:
        """
        Apply share bonus to a user.

        Args:
            uid: User ID
            today: Date to check (defaults to today)
            base_b: Base daily bonus amount (defaults to reading from state)

        Returns:
            Dict with status and bonus details
        """
        if today is None:
            today = date.today()

        # Read user wallet to check if already awarded today
        wallet = self.store.read_user_wallet(uid)

        if wallet.last_share_bonus_date == today:
            return {
                "status": "already_awarded",
                "uid": uid,
                "message": "Share bonus already awarded today.",
            }

        # Get base_b from daily bonus state if not provided
        if base_b is None:
            base_b = self.store.read_last_daily_b()

        if base_b.value <= 0:
            return {
                "status": "no_bonus_available",
                "uid": uid,
                "message": "No daily bonus configured yet.",
            }

        # Compute share bonus
        share_bonus = self.compute_share_bonus(base_b)

        if share_bonus.value <= 0:
            return {
                "status": "zero_bonus",
                "uid": uid,
                "message": "Computed share bonus is zero.",
            }

        # Apply the bonus
        applied = self.store.apply_share_bonus(uid, share_bonus, today)

        if applied:
            return {
                "status": "success",
                "uid": uid,
                "bonus_sc": share_bonus.to_str(),
                "base_b_sc": base_b.to_str(),
                "share_ratio": str(self.cfg.share_ratio),
            }
        else:
            return {
                "status": "already_awarded",
                "uid": uid,
                "message": "Share bonus already awarded today (race condition).",
            }


class ShareBonusServiceFactory:
    """Factory for creating ShareBonusService instances."""

    @staticmethod
    def create(store: ShareBonusStore, cfg: "MonetizationConfig") -> ShareBonusService:
        return ShareBonusService(store, cfg)
