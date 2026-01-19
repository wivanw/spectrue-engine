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

import os

PRODUCTS: Dict[str, ProductConfig] = {
    # Plans
    "plan_base": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_PLAN_BASE", ""),
        type=ProductType.PLAN,
        user_grant_sc=Decimal("300"),
        pool_contribution_sc=Decimal("160"),
    ),
    "plan_pro": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_PLAN_PRO", ""),
        type=ProductType.PLAN,
        user_grant_sc=Decimal("700"),
        pool_contribution_sc=Decimal("240"),
    ),
    "plan_max": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_PLAN_MAX", ""),
        type=ProductType.PLAN,
        user_grant_sc=Decimal("2400"),
        pool_contribution_sc=Decimal("480"),
    ),
    "plan_enterprise": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_PLAN_ENTERPRISE", ""),
        type=ProductType.PLAN,
        user_grant_sc=Decimal("5100"),
        pool_contribution_sc=Decimal("720"),
    ),
    # Packs
    "pack_5": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_PACK_SMALL", ""),
        type=ProductType.PACK,
        user_grant_sc=Decimal("250"),
        pool_contribution_sc=Decimal("200"),
    ),
    "pack_10": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_PACK_MEDIUM", ""),
        type=ProductType.PACK,
        user_grant_sc=Decimal("600"),
        pool_contribution_sc=Decimal("320"),
    ),
    "pack_30": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_PACK_LARGE", ""),
        type=ProductType.PACK,
        user_grant_sc=Decimal("2100"),
        pool_contribution_sc=Decimal("720"),
    ),
    "pack_60": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_PACK_GIANT", ""),
        type=ProductType.PACK,
        user_grant_sc=Decimal("4800"),
        pool_contribution_sc=Decimal("960"),
    ),
    # Donations
    "donation_5": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_DONATE_TINY", ""),
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("400"),
    ),
    "donation_10": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_DONATE_SMALL", ""),
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("800"),
    ),
    "donation_50": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_DONATE_MEDIUM", ""),
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("4000"),
    ),
    "donation_100": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_DONATE_LARGE", ""),
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("8000"),
    ),
    "donation_1000": ProductConfig(
        stripe_price_id=os.getenv("STRIPE_PRICE_DONATE_HERO", ""),
        type=ProductType.DONATION,
        user_grant_sc=Decimal("0"),
        pool_contribution_sc=Decimal("80000"),
    ),
}

# Reverse Index for Stripe ID lookup
STRIPE_ID_TO_KEY: Dict[str, str] = {
    p.stripe_price_id: k
    for k, p in PRODUCTS.items()
    if p.stripe_price_id
}

def get_product_by_key(key: str) -> Optional[ProductConfig]:
    cfg = PRODUCTS.get(key)
    if not cfg or not cfg.stripe_price_id:
        return None
    return cfg

def get_product_by_stripe_id(stripe_id: str) -> Optional[ProductConfig]:
    if not stripe_id:
        return None
    key = STRIPE_ID_TO_KEY.get(stripe_id)
    if key:
        return PRODUCTS.get(key)
    return None