from decimal import Decimal
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BonusAlgoConfig:
    bonus_buffer_days: int = 30
    daily_spend_cap_ratio: Decimal = Decimal("0.03")
    alpha_budget: Decimal = Decimal("0.1") # EMA smoothing factor
    active_users_floor: int = 100
    max_bonus_up_ratio: Decimal = Decimal("0.15")
    max_bonus_down_ratio: Decimal = Decimal("0.10")
    b_min_sc: Decimal = Decimal("0.01")
    b_max_sc: Decimal = Decimal("10.0")

class DailyBonusCalculator:
    def __init__(self, config: BonusAlgoConfig):
        self.config = config

    def calculate_daily_bonus(
        self, 
        pool_available: Decimal, 
        active_users: int, 
        prev_B: Decimal, 
        prev_b: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculates new Total Budget (B) and Per-User Bonus (b).
        Returns (B_new, b_new).
        """
        # 1. Target Budget
        if self.config.bonus_buffer_days > 0:
            B_target = pool_available / self.config.bonus_buffer_days
        else:
            B_target = pool_available

        # 2. Cap
        B_cap = pool_available * self.config.daily_spend_cap_ratio
        B_raw = min(B_target, B_cap)

        # 3. Smoothing (EMA)
        if prev_B == Decimal("0.0"):
            B_new = B_raw
        else:
            B_new = (self.config.alpha_budget * B_raw) + \
                    ((Decimal("1.0") - self.config.alpha_budget) * prev_B)

        # 4. Per User Calculation
        user_divisor = max(active_users, self.config.active_users_floor)
        b_raw = B_new / user_divisor

        # 5. Rate Limiting (vs prev_b)
        if prev_b > Decimal("0.0"):
            max_up = prev_b * (Decimal("1.0") + self.config.max_bonus_up_ratio)
            max_down = prev_b * (Decimal("1.0") - self.config.max_bonus_down_ratio)

            if b_raw > max_up:
                b_raw = max_up
            elif b_raw < max_down:
                b_raw = max_down

        # 6. Clamping
        b_final = max(self.config.b_min_sc, min(b_raw, self.config.b_max_sc))

        # 7. Solvency Check
        # Total cost = b_final * active_users
        total_cost = b_final * active_users
        if total_cost > pool_available:
            # Scale down to fit pool exactly
            if active_users > 0:
                b_final = pool_available / active_users
            else:
                b_final = Decimal("0.0")

        return B_new, b_final
