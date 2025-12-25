from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class User:
    uid: str
    email: Optional[str] = None
    balance_sc: Decimal = Decimal("0.0")
    last_seen_at: Optional[datetime] = None
    last_daily_bonus_date: Optional[str] = None # YYYY-MM-DD
    last_share_bonus_date: Optional[str] = None # YYYY-MM-DD
    plan_tier: str = "free"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        return cls(
            uid=data.get("uid", ""),
            email=data.get("email"),
            balance_sc=Decimal(str(data.get("balance_sc", 0.0))),
            last_seen_at=data.get("last_seen_at"),
            last_daily_bonus_date=data.get("last_daily_bonus_date"),
            last_share_bonus_date=data.get("last_share_bonus_date"),
            plan_tier=data.get("plan_tier", "free")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "email": self.email,
            "balance_sc": float(self.balance_sc),
            "last_seen_at": self.last_seen_at,
            "last_daily_bonus_date": self.last_daily_bonus_date,
            "last_share_bonus_date": self.last_share_bonus_date,
            "plan_tier": self.plan_tier
        }
