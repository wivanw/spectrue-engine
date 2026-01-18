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
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
import hashlib

from spectrue_core.monetization.types import MoneySC


class LedgerEntryType(str, Enum):
    RESERVE = "reserve"
    SETTLE = "settle"
    RELEASE = "release"
    DEPOSIT = "deposit"
    ADJUSTMENT = "adjustment"
    CHARGE = "charge"        # Standard charge type
    BONUS = "bonus"          # Bonus ledger type


class LedgerEntryStatus(str, Enum):
    PENDING = "pending"
    SETTLED = "settled"
    REVERSED = "reversed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    # Core fields
    idempotency_key: str
    entry_type: LedgerEntryType
    amount_sc: MoneySC
    status: LedgerEntryStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Optional context
    event_id: str | None = None
    user_id: str | None = None
    run_id: str | None = None
    
    # Charge-specific fields (Optional)
    split: Optional[Dict[str, str]] = None      # {"take_available_sc": "...", "take_balance_sc": "..."}
    new_wallet: Optional[Dict[str, str]] = None # {"available_sc": "...", "balance_sc": "..."}
    reason: Optional[str] = None
    
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "idempotency_key": self.idempotency_key,
            "entry_type": self.entry_type.value,
            "amount_sc": str(self.amount_sc),
            "status": self.status.value,
            "created_at": self.created_at,
            "event_id": self.event_id,
            "user_id": self.user_id,
            "run_id": self.run_id,
            "split": self.split,
            "new_wallet": self.new_wallet,
            "reason": self.reason,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        """Load from dictionary."""
        # Handle datetime parsing if it's a string
        created = data.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        
        return cls(
            idempotency_key=data.get("idempotency_key", ""),
            entry_type=LedgerEntryType(data.get("entry_type", LedgerEntryType.RESERVE.value)),
            amount_sc=MoneySC(data.get("amount_sc", "0")),
            status=LedgerEntryStatus(data.get("status", LedgerEntryStatus.PENDING.value)),
            created_at=created or datetime.utcnow(),
            event_id=data.get("event_id"),
            user_id=data.get("user_id"),
            run_id=data.get("run_id"),
            split=data.get("split"),
            new_wallet=data.get("new_wallet"),
            reason=data.get("reason"),
            meta=data.get("meta") or {},
        )


def build_idempotency_key(*parts: str) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return ":".join(cleaned)

def build_charge_idempotency_key(uid: str, run_id: str) -> str:
    """
    Build a stable idempotency key for a charge.
    Format: {uid}:{run_id}:charge:v1
    """
    raw = f"{uid}:{run_id}:charge:v1"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
