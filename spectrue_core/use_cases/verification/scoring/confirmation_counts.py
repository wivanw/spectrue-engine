# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from typing import Any


def compute_confirmation_counts(corroboration: dict[str, Any] | None, *, lam: float = 0.35) -> dict[str, float]:
    """
    Deterministic confirmation counters.
    - C_precise: unique publishers with direct anchors for SUPPORT/REFUTE
    - C_corr: near-dup corroboration clusters (weak corroboration)
    - C_total: blended count
    """
    if not isinstance(corroboration, dict):
        return {"C_precise": 0.0, "C_corr": 0.0, "C_total": 0.0}

    ps = float(corroboration.get("precision_publishers_support", 0) or 0)
    pr = float(corroboration.get("precision_publishers_refute", 0) or 0)
    cs = float(corroboration.get("corroboration_clusters_support", 0) or 0)
    cr = float(corroboration.get("corroboration_clusters_refute", 0) or 0)

    # We expose both sides; C_total can be interpreted by consumers.
    C_precise = max(ps, pr)
    C_corr = max(cs, cr)
    C_total = C_precise + lam * C_corr
    return {"C_precise": C_precise, "C_corr": C_corr, "C_total": C_total}
