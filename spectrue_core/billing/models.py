from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, Optional
from datetime import datetime

@dataclass
class FreeSubsidyPool:
    available_balance_sc: Decimal = Decimal("0.0")
    last_updated: datetime = field(default_factory=datetime.utcnow)
    # Key: YYYY-MM-DD string, Value: Amount to be released
    locked_buckets: Dict[str, Decimal] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreeSubsidyPool":
        return cls(
            available_balance_sc=Decimal(str(data.get("available_balance_sc", 0.0))),
            last_updated=data.get("last_updated", datetime.utcnow()),
            locked_buckets={k: Decimal(str(v)) for k, v in data.get("locked_buckets", {}).items()}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_balance_sc": float(self.available_balance_sc),
            "last_updated": self.last_updated,
            "locked_buckets": {k: float(v) for k, v in self.locked_buckets.items()}
        }

@dataclass
class DailyBonusState:
    last_run_date: str = "" # YYYY-MM-DD
    smoothed_budget_B: Decimal = Decimal("0.0")
    last_user_bonus_b: Decimal = Decimal("0.0")
    last_run_stats: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DailyBonusState":
        return cls(
            last_run_date=data.get("last_run_date", ""),
            smoothed_budget_B=Decimal(str(data.get("smoothed_budget_B", 0.0))),
            last_user_bonus_b=Decimal(str(data.get("last_user_bonus_b", 0.0))),
            last_run_stats=data.get("last_run_stats", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_run_date": self.last_run_date,
            "smoothed_budget_B": float(self.smoothed_budget_B),
            "last_user_bonus_b": float(self.last_user_bonus_b),
            "last_run_stats": self.last_run_stats
        }
