# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Single claim priority (M146 T022).

Priority used to be encoded as importance, check_worthiness and harm_potential
with four different harm normalizations and an EVOI-vs-constant-cost cut that
did not share a scale with its inputs. Now:

    value(claim) = checkability × importance × harm_norm × uncertainty

- checkability: 0 when the claim is not verifiable by nature, else 1 (the P3
  Decisions sensor replaces this with P(verification_target ≠ none));
- importance: [0, 1];
- harm_norm = harm_potential / 5 (1–5 scale → 0.2–1.0; no factual claim gets 0);
- uncertainty = entropy(prior); a graph uncertainty, when present, raises it.

Target selection is then a unit-cost knapsack: take the highest-value claims
with value > 0 up to the budget class ceiling. No marginal-cost constant.
"""

from __future__ import annotations

import math
from typing import Any

from spectrue_core.domain.claims import fields
from spectrue_core.domain.verification.verdict.global_aggregation import is_unverifiable_by_nature

# Ordering prior when the extractor gave no importance / harm (explicit, documented).
IMPORTANCE_IF_ABSENT = 0.5
HARM_IF_ABSENT = 1.0  # lowest stakes on the 1–5 scale


def harm_norm(claim: Any) -> float:
    h = fields.or_default(fields.number(claim, "harm_potential"), HARM_IF_ABSENT)
    return max(0.0, min(1.0, h / 5.0))


def checkability(claim: Any) -> float:
    return 0.0 if is_unverifiable_by_nature(claim if isinstance(claim, dict) else {}) else 1.0


def _entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def uncertainty(claim: Any, prior_p: float = 0.5) -> float:
    base = _entropy(prior_p)
    gu = fields.unit(claim, "graph_uncertainty")
    return base if gu is None else max(base * (0.5 + 0.5 * gu), 0.0)


def claim_value(claim: Any, *, prior_p: float = 0.5, value_uncertainty: float = 1.0) -> float:
    importance = fields.or_default(fields.unit(claim, "importance"), IMPORTANCE_IF_ABSENT)
    return value_uncertainty * checkability(claim) * importance * harm_norm(claim) * uncertainty(claim, prior_p)
