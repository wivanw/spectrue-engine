# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class User:
    uid: str
    email: Optional[str] = None
    balance_sc: Decimal = Decimal("0.0")
    available_sc: Decimal = Decimal("0.0")  # V3: bonus credits
    last_seen_at: Optional[datetime] = None
    last_daily_bonus_date: Optional[str] = None  # YYYY-MM-DD
    last_share_bonus_date: Optional[str] = None  # YYYY-MM-DD
    plan_tier: str = "free"

    @property
    def total_sc(self) -> Decimal:
        """Total credits (paid + bonus)."""
        return self.balance_sc + self.available_sc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        raw_balance = data.get("balance_sc", 0)
        raw_available = data.get("available_sc", 0)
        try:
            balance = Decimal(str(raw_balance))
        except Exception:
            balance = Decimal("0")
        try:
            available = Decimal(str(raw_available))
        except Exception:
            available = Decimal("0")
        return cls(
            uid=data.get("uid", ""),
            email=data.get("email"),
            balance_sc=balance,
            available_sc=available,
            last_seen_at=data.get("last_seen_at"),
            last_daily_bonus_date=data.get("last_daily_bonus_date"),
            last_share_bonus_date=data.get("last_share_bonus_date"),
            plan_tier=data.get("plan_tier", "free")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "email": self.email,
            "balance_sc": str(self.balance_sc),
            "available_sc": str(self.available_sc),
            "last_seen_at": self.last_seen_at,
            "last_daily_bonus_date": self.last_daily_bonus_date,
            "last_share_bonus_date": self.last_share_bonus_date,
            "plan_tier": self.plan_tier
        }

