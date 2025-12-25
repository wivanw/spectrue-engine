from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional

class ProductType(Enum):
    PLAN = "plan"
    PACK = "pack"
    DONATION = "donation"

@dataclass
class ProductConfig:
    stripe_price_id: str
    type: ProductType
    user_grant_sc: Decimal
    pool_contribution_sc: Decimal

# Global Config
OPERATIONAL_LOCK_RATIO = Decimal("0.5") # Default, adjust as needed

PRODUCTS: Dict[str, ProductConfig] = {
    # Plans
    "plan_base": ProductConfig(
        stripe_price_id="price_plan_base",
        type=ProductType.PLAN,
        user_grant_sc=Decimal("300"),
        pool_contribution_sc=Decimal("160"),
    ),
    "plan_pro": ProductConfig(
        stripe_price_id="price_plan_pro",
        type=ProductType.PLAN,
        user_grant_sc=Decimal("700"),
        pool_contribution_sc=Decimal("240"),
    ),
    "plan_max": ProductConfig(
        stripe_price_id="price_plan_max",
        type=ProductType.PLAN,
        user_grant_sc=Decimal("2400"),
        pool_contribution_sc=Decimal("480"),
    ),
    "plan_enterprise": ProductConfig(
        stripe_price_id="price_plan_enterprise",
        type=ProductType.PLAN,
        user_grant_sc=Decimal("5100"),
        pool_contribution_sc=Decimal("720"),
    ),
    # Packs
    "pack_5": ProductConfig(
        stripe_price_id="price_pack_5",
        type=ProductType.PACK,
        user_grant_sc=Decimal("250"),
        pool_contribution_sc=Decimal("200"),
    ),
    "pack_10": ProductConfig(
        stripe_price_id="price_pack_10",
        type=ProductType.PACK,
        user_grant_sc=Decimal("600"),
        pool_contribution_sc=Decimal("320"),
    ),
    "pack_30": ProductConfig(
        stripe_price_id="price_pack_30",
        type=ProductType.PACK,
        user_grant_sc=Decimal("2100"),
        pool_contribution_sc=Decimal("720"),
    ),
    "pack_60": ProductConfig(
        stripe_price_id="price_pack_60",
        type=ProductType.PACK,
        user_grant_sc=Decimal("4800"),
        pool_contribution_sc=Decimal("960"),
    ),
    # Donations
    "donation_5": ProductConfig(
        stripe_price_id="price_donation_5",
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("400"),
    ),
    "donation_10": ProductConfig(
        stripe_price_id="price_donation_10",
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("800"),
    ),
    "donation_50": ProductConfig(
        stripe_price_id="price_donation_50",
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("4000"),
    ),
    "donation_100": ProductConfig(
        stripe_price_id="price_donation_100",
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("8000"),
    ),
    "donation_1000": ProductConfig(
        stripe_price_id="price_donation_1000",
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("80000"),
    ),
}

def get_product_by_key(key: str) -> Optional[ProductConfig]:
    return PRODUCTS.get(key)