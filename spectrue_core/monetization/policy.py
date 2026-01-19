# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from spectrue_core.monetization.types import EligibilityAllowance, MoneySC


@dataclass(frozen=True, slots=True)
class UserEligibility:
    eligible: bool
    daily_remaining_sc: MoneySC
    monthly_remaining_sc: MoneySC
    reason: str | None = None

    def to_allowance(self) -> EligibilityAllowance:
        return EligibilityAllowance(
            eligible=self.eligible,
            daily_remaining_sc=self.daily_remaining_sc,
            monthly_remaining_sc=self.monthly_remaining_sc,
            reason=self.reason,
        )


@runtime_checkable
class BillingPolicy(Protocol):
    def get_user_eligibility(self, user_id: str) -> UserEligibility:
        ...
