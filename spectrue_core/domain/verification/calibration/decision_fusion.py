# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Bayesian fusion of Decisions API sensor readings (M145 R028).

The Decisions model is a noisy sensor (Constitution II). A reading `p` enters
deterministic code only as calibrated evidence in log-odds:

    delta(p) = weight * logit(p) + bias

`weight` / `bias` are fitted offline by logistic regression of reference labels
on logit(p), with the existing signal's logit as offset (Phase 4). With
`weight = bias = 0` (uncalibrated) the reading carries no information and every
function here leaves the prior unchanged. There are no thresholds here: each
caller compares the posterior with the decision rule its step already uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-6


def logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, float(p)))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class SensorCalibration:
    """Fitted sensor model for one gate (see `DecisionSensorCalibration` in config)."""

    weight: float = 0.0
    bias: float = 0.0
    # Measured error rate of the reference path this sensor can replace (G5).
    reference_error_rate: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.weight != 0.0 or self.bias != 0.0

    def delta(self, p: float) -> float:
        """Log-likelihood-ratio contribution of a reading (0 when uncalibrated)."""
        return self.weight * logit(p) + self.bias if self.calibrated else 0.0


def fuse(prior: float, reading: float, cal: SensorCalibration) -> float:
    """Posterior after one sensor reading: sigmoid(logit(prior) + delta(reading))."""
    if not cal.calibrated:
        return float(prior)
    return sigmoid(logit(prior) + cal.delta(reading))


def calibrated_probability(reading: float, cal: SensorCalibration) -> float | None:
    """P(event | reading) from the sensor alone; None when uncalibrated."""
    return sigmoid(cal.delta(reading)) if cal.calibrated else None


def any_of(probabilities: list[float]) -> float:
    """P(at least one event) for independent events: 1 - prod(1 - p_i)."""
    out = 1.0
    for p in probabilities:
        out *= 1.0 - min(1.0, max(0.0, float(p)))
    return 1.0 - out


def expected_error(q: float) -> float:
    """Expected error of the MAP label of a calibrated binary probability."""
    return min(q, 1.0 - q)
