# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
from __future__ import annotations

import dataclasses
import datetime
import enum
import inspect
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="SchemaModel")


class SchemaModel(BaseModel):
    """
    Canonical base for schema models (Pydantic v2).

    - Ignores extra fields for backward compatibility.
    - Provides `to_dict()` / `from_dict()` for consistent serialization.
    """

    model_config = {"extra": "ignore"}

    def to_dict(self) -> dict[str, Any]:
        return dump_schema(self)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        return load_schema(cls, data)


def dump_schema(model: Any) -> dict[str, Any]:
    """
    Dump a schema model to a JSON-safe dict.

    For Pydantic v2 models, uses `model_dump(mode="json")`.
    """
    if isinstance(model, BaseModel):
        return model.model_dump(mode="json", exclude_none=True)
    if dataclasses.is_dataclass(model):
        return _json_safe(dataclasses.asdict(model))
    if hasattr(model, "to_dict") and callable(getattr(model, "to_dict")):
        out = model.to_dict()
        return _json_safe(out if isinstance(out, dict) else {"value": out})
    raise TypeError(f"Unsupported schema type for dump: {type(model)!r}")


def load_schema(model_cls: type[T], data: dict[str, Any]) -> T:
    """
    Load a schema model from a dict.

    For Pydantic v2 models, uses `model_validate`.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Schema input must be a dict, got: {type(data)!r}")
    if inspect.isclass(model_cls) and issubclass(model_cls, BaseModel):
        return model_cls.model_validate(data)
    if hasattr(model_cls, "from_dict") and callable(getattr(model_cls, "from_dict")):
        return model_cls.from_dict(data)  # type: ignore[no-any-return]
    if dataclasses.is_dataclass(model_cls):
        return model_cls(**data)  # type: ignore[misc]
    raise TypeError(f"Unsupported schema type for load: {model_cls!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value
