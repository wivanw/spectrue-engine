# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Decisions API sensor reader (M145 Phase R).

Decisions is a noisy sensor (Constitution II): this module only *reads* it.
Deterministic code in `use_cases/retrieval/decision_gates.py` turns readings into
decisions via the step's own Bayesian / EVOI rule. Per gate mode:

  off    -> no call; returns None.
  shadow -> call billed `internal` (never charged to the user); the reading is
            returned with `applied=False` so callers only trace it.
  on     -> call billed to the run; `applied=True`.

Provider failures and refusals return None with a typed trace event
(`decision_gate.non_ok`); the caller then keeps today's behaviour. Any other
exception is a bug and propagates (Constitution III).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

from spectrue_core.llm.decisions import DecisionQuestion, DecisionResult
from spectrue_core.llm.errors import LLMCallError, LLMFailureKind
from spectrue_core.runtime_config import DecisionsConfig
from spectrue_core.utils.trace import Trace

GATE_MODES = ("off", "shadow", "on")

# Failures of the external call that become a typed trace outcome.
_PROVIDER_ERRORS: tuple[type[BaseException], ...] = (LLMCallError, asyncio.TimeoutError)


@dataclass(frozen=True)
class SensorReading:
    result: DecisionResult
    applied: bool  # False in shadow: trace only, never change the pipeline
    latency_ms: int

    def probability(self, name: str) -> float:
        """Predicate probability (refusals never reach here: see read_sensor)."""
        a = self.result.get(name)
        if a.probability is None:
            raise LLMCallError(message=f"answer {name!r} has no probability",
                               kind=LLMFailureKind.SCHEMA_VALIDATION_FAILED)
        return a.probability


def gate_mode(cfg: DecisionsConfig, gate: str) -> str:
    """Validated mode for a gate. A non-mode value is a wiring bug and raises."""
    mode = cfg.mode(gate)
    if mode not in GATE_MODES:
        raise ValueError(f"Invalid Decisions gate mode for {gate!r}: {mode!r}")
    return mode


def _error_kind(exc: BaseException) -> str:
    return exc.kind.value if isinstance(exc, LLMCallError) else "timeout"


async def read_sensor(
    *,
    gate: str,
    cfg: DecisionsConfig,
    llm_client: Any,
    input_text: str,
    questions: list[DecisionQuestion],
    trace_meta: dict[str, Any] | None = None,
    independent: bool = False,
) -> SensorReading | None:
    """One Decisions request for `gate`, honouring its mode. None = no usable reading.

    `independent`: the questions feed separate fields, so a refused question drops
    only its own answer (the caller keeps that field's current value). Otherwise
    one refusal makes the whole reading unusable.
    """
    mode = gate_mode(cfg, gate)
    if mode == "off":
        return None
    if mode == "shadow" and cfg.shadow_sample_rate < 1.0 and random.random() >= cfg.shadow_sample_rate:
        return None
    meta = {"gate": gate, "mode": mode, **(trace_meta or {})}
    if len(input_text) > cfg.max_input_chars:
        # Never silently truncate: the reference path sees the full input.
        Trace.event("decision_gate.non_ok", {**meta, "outcome": LLMFailureKind.INPUT_TOO_LONG.value})
        return None

    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            llm_client.call_decision(
                model=cfg.model,
                input=input_text,
                questions=questions,
                timeout=cfg.timeout_sec,
                trace_kind=f"decision_gate.{gate}",
                stage=f"decision_gate.{gate}",
                billing="user" if mode == "on" else "internal",
            ),
            timeout=cfg.timeout_sec + 1.0,
        )
    except _PROVIDER_ERRORS as exc:
        Trace.event("decision_gate.non_ok", {**meta, "outcome": _error_kind(exc), "error": str(exc)[:200]})
        return None
    latency_ms = int((time.monotonic() - t0) * 1000)

    refused = [n for n, a in result.answers.items() if a.is_refusal]
    if refused:
        # A refusal is never mapped to a value.
        Trace.event("decision_gate.non_ok", {**meta, "outcome": "refusal", "questions": refused})
        kept = {n: a for n, a in result.answers.items() if not a.is_refusal}
        if not independent or not kept:
            return None
        result = DecisionResult(answers=kept, model=result.model, usage=result.usage)
    Trace.event("decision_gate.reading", {
        **meta,
        "latency_ms": latency_ms,
        "input_tokens": int(result.usage.get("input_tokens") or 0),
        "answers": {
            n: {"type": a.type, "probability": a.probability, "choice": a.choice,
                "probabilities": a.probabilities, "confidence": a.confidence}
            for n, a in result.answers.items()
        },
    })
    return SensorReading(result=result, applied=mode == "on", latency_ms=latency_ms)
