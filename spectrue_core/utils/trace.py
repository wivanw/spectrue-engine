# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spectrue Engine is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Spectrue Engine. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import contextvars
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectrue_core.utils.runtime import is_local_run

_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("spectrue_trace_id", default=None)
_trace_enabled_var: contextvars.ContextVar[bool] = contextvars.ContextVar("spectrue_trace_enabled", default=False)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _redact_text(s: str) -> str:
    if not s:
        return s

    # URL params
    s = re.sub(r"([?&]key=)[^&]+", r"\1***", s, flags=re.IGNORECASE)
    s = re.sub(r"([?&]api_key=)[^&]+", r"\1***", s, flags=re.IGNORECASE)
    s = re.sub(r"([?&]access_token=)[^&]+", r"\1***", s, flags=re.IGNORECASE)

    # Bearer tokens / auth headers
    s = re.sub(r"(Authorization:\s*Bearer\s+)[^\s]+", r"\1***", s, flags=re.IGNORECASE)
    s = re.sub(r"(Bearer\s+)[A-Za-z0-9._-]+", r"\1***", s)

    return s


def _sanitize(obj: Any, *, max_str: int = 4000, max_list: int = 100, max_dict: int = 200) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        s = _redact_text(obj)
        if len(s) > max_str:
            return s[:max_str] + "…"
        return s
    if isinstance(obj, bytes):
        return f"<bytes:{len(obj)}>"
    if isinstance(obj, (list, tuple)):
        out = [_sanitize(x, max_str=max_str, max_list=max_list, max_dict=max_dict) for x in obj[:max_list]]
        if len(obj) > max_list:
            out.append(f"...(+{len(obj) - max_list} more)")
        return out
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_dict:
                out["..."] = f"(+{len(obj) - max_dict} more keys)"
                break
            key = str(k)
            if key.lower() in ("authorization", "api_key", "key", "openai_api_key", "tavily_api_key"):
                out[key] = "***"
            else:
                out[key] = _sanitize(v, max_str=max_str, max_list=max_list, max_dict=max_dict)
        return out
    return _sanitize(str(obj), max_str=max_str, max_list=max_list, max_dict=max_dict)


def _trace_dir() -> Path:
    # Keep traces next to other local artifacts (cache/), but never rely on global paths.
    p = Path("data/trace")
    p.mkdir(parents=True, exist_ok=True)
    return p


def trace_enabled() -> bool:
    return bool(_trace_enabled_var.get())


def current_trace_id() -> str | None:
    return _trace_id_var.get()


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    enabled: bool


class Trace:
    """
    Local-only trace sink (JSONL) for debugging.
    Never enabled in production by default.
    """

    @staticmethod
    def start(trace_id: str) -> TraceContext:
        enabled = is_local_run() and os.getenv("SPECTRUE_TRACE_DISABLE", "").strip().lower() not in (
            "1",
            "true",
            "yes",
            "on",
        )
        _trace_id_var.set(trace_id)
        _trace_enabled_var.set(enabled)
        if enabled:
            Trace.event("trace.start", {"trace_id": trace_id})
        return TraceContext(trace_id=trace_id, enabled=enabled)

    @staticmethod
    def stop() -> None:
        tid = current_trace_id()
        if trace_enabled() and tid:
            Trace.event("trace.stop", {"trace_id": tid})
        _trace_enabled_var.set(False)
        _trace_id_var.set(None)

    @staticmethod
    def event(name: str, data: Any | None = None) -> None:
        if not trace_enabled():
            return
        tid = current_trace_id()
        if not tid:
            return

        rec = {
            "ts_ms": _now_ms(),
            "trace_id": tid,
            "event": str(name),
            "data": _sanitize(data),
        }
        try:
            path = _trace_dir() / f"{tid}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # Tracing must never break the main flow.
            return

