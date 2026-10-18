# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Typed request/response model for the OpenAI Decisions API (POST /v1/decisions).

A decision call returns probabilities over questions whose answer shape is fixed
in advance (predicate / choice / score). No text is generated.

Contract (Constitution III): a `refusal` answer is a first-class result, never
mapped to a label. Callers decide whether to escalate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from spectrue_core.llm.errors import LLMCallError, LLMFailureKind

AnswerType = Literal["predicate", "choice", "score", "refusal"]


@dataclass(frozen=True)
class DecisionQuestion:
    """One typed question. Build with the classmethods below."""

    name: str
    payload: dict[str, Any]

    @classmethod
    def predicate(cls, name: str, instructions: str) -> "DecisionQuestion":
        return cls(name, {"type": "predicate", "name": name, "instructions": instructions})

    @classmethod
    def choice(cls, name: str, instructions: str, choices: dict[str, str]) -> "DecisionQuestion":
        return cls(name, {
            "type": "choice",
            "name": name,
            "instructions": instructions,
            "choices": [{"value": v, "description": d} for v, d in choices.items()],
        })

    @classmethod
    def score(cls, name: str, instructions: str, levels: list[tuple[str, str]]) -> "DecisionQuestion":
        """`levels` are (label, description), ordered lowest to highest."""
        return cls(name, {
            "type": "score",
            "name": name,
            "instructions": instructions,
            "levels": [{"label": label, "description": d} for label, d in levels],
        })


@dataclass(frozen=True)
class DecisionAnswer:
    name: str
    type: AnswerType
    probability: float | None = None  # predicate
    choice: str | None = None  # choice
    score: float | None = None  # score
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None

    @property
    def is_refusal(self) -> bool:
        return self.type == "refusal"

    @property
    def top_confidence(self) -> float | None:
        """Comparable confidence across answer types (None for refusals)."""
        if self.type == "predicate" and self.probability is not None:
            return max(self.probability, 1.0 - self.probability)
        return self.confidence


@dataclass(frozen=True)
class DecisionResult:
    answers: dict[str, DecisionAnswer]
    model: str
    usage: dict[str, Any]

    def get(self, name: str) -> DecisionAnswer:
        try:
            return self.answers[name]
        except KeyError:
            raise LLMCallError(
                message=f"Decision response missing answer '{name}'",
                kind=LLMFailureKind.SCHEMA_VALIDATION_FAILED,
            ) from None


def _prob(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMCallError(message=f"Decision {where} is not a number", kind=LLMFailureKind.SCHEMA_VALIDATION_FAILED)
    f = float(value)
    if not 0.0 <= f <= 1.0:
        raise LLMCallError(message=f"Decision {where} out of range: {f}", kind=LLMFailureKind.SCHEMA_VALIDATION_FAILED)
    return f


def parse_decision_response(raw: Any, questions: list[DecisionQuestion]) -> DecisionResult:
    """Validate a raw /v1/decisions JSON body. Raises LLMCallError on any shape problem."""
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), list):
        raise LLMCallError(message="Decision response has no answers list", kind=LLMFailureKind.SCHEMA_VALIDATION_FAILED)

    expected = {q.name: q.payload["type"] for q in questions}
    answers: dict[str, DecisionAnswer] = {}
    for item in raw["answers"]:
        if not isinstance(item, dict):
            continue
        name, typ = item.get("name"), item.get("type")
        if name not in expected:
            continue
        if typ == "refusal":
            answers[name] = DecisionAnswer(name=name, type="refusal")
            continue
        if typ != expected[name]:
            raise LLMCallError(
                message=f"Decision answer '{name}' has type {typ!r}, expected {expected[name]!r}",
                kind=LLMFailureKind.SCHEMA_VALIDATION_FAILED,
            )
        probs = {
            str(p.get("value", p.get("label"))): _prob(p.get("probability"), f"{name}.probabilities")
            for p in item.get("probabilities") or []
            if isinstance(p, dict)
        }
        conf = item.get("confidence")
        answers[name] = DecisionAnswer(
            name=name,
            type=typ,
            probability=_prob(item.get("probability"), f"{name}.probability") if typ == "predicate" else None,
            choice=str(item["choice"]) if typ == "choice" and item.get("choice") is not None else None,
            score=float(item["score"]) if typ == "score" and isinstance(item.get("score"), (int, float)) else None,
            probabilities=probs,
            confidence=_prob(conf, f"{name}.confidence") if conf is not None else None,
        )
        if typ == "choice" and answers[name].choice is None:
            raise LLMCallError(message=f"Decision answer '{name}' has no choice", kind=LLMFailureKind.SCHEMA_VALIDATION_FAILED)

    missing = set(expected) - set(answers)
    if missing:
        raise LLMCallError(
            message=f"Decision response missing answers: {sorted(missing)}",
            kind=LLMFailureKind.SCHEMA_VALIDATION_FAILED,
        )
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    return DecisionResult(answers=answers, model=str(raw.get("model") or ""), usage=usage)
