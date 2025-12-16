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
import re
import time
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from spectrue_core.utils.runtime import is_local_run
from spectrue_core.runtime_config import EngineRuntimeConfig

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
        if len(s) <= max_str:
            return s
        head_tail_len = min(300, max_str // 2)
        return {
            "len": len(s),
            "sha256": hashlib.sha256(s.encode("utf-8")).hexdigest(),
            "head": s[:head_tail_len],
            "tail": s[-head_tail_len:] if head_tail_len else "",
        }
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
    def start(trace_id: str, *, runtime: EngineRuntimeConfig | None = None) -> TraceContext:
        # Centralized config: trace is local-only and can be disabled via runtime.features.trace_enabled.
        runtime = runtime or EngineRuntimeConfig.load_from_env()
        enabled = bool(is_local_run() and runtime.features.trace_enabled)
        _trace_id_var.set(trace_id)
        _trace_enabled_var.set(enabled)
        if enabled:
            # M47: Clean up old traces, keep only the latest one
            try:
                trace_dir = _trace_dir()
                for old_file in trace_dir.glob("*.jsonl"):
                    try:
                        old_file.unlink()
                    except Exception:
                        pass
            except Exception:
                pass
            
            Trace.event("trace.start", {
                "trace_id": trace_id,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
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
            # M53: Sanitize trace_id for filename safety (e.g. replace ':' with '_')
            safe_tid = "".join(c if c.isalnum() or c in "._-" else "_" for c in tid)
            path = _trace_dir() / f"{safe_tid}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # Tracing must never break the main flow.
            return
