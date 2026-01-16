# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import contextvars
import json
import re
import time
from dataclasses import dataclass
import hashlib
from pathlib import Path
from contextlib import contextmanager
from typing import Any
import os

from spectrue_core.utils.runtime import is_local_run
from spectrue_core.runtime_config import EngineRuntimeConfig

_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("spectrue_trace_id", default=None)
_trace_enabled_var: contextvars.ContextVar[bool] = contextvars.ContextVar("spectrue_trace_enabled", default=False)
_redact_pii_enabled_var: contextvars.ContextVar[bool] = contextvars.ContextVar("spectrue_redact_pii", default=False)
_trace_safe_payloads_var: contextvars.ContextVar[bool] = contextvars.ContextVar("spectrue_trace_safe_payloads", default=True)
_trace_max_head_var: contextvars.ContextVar[int] = contextvars.ContextVar("spectrue_trace_max_head", default=120)
_trace_max_inline_var: contextvars.ContextVar[int] = contextvars.ContextVar("spectrue_trace_max_inline", default=600)
_trace_max_override_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("spectrue_trace_max_override", default=None)
_trace_phase_starts: contextvars.ContextVar[dict[str, int]] = contextvars.ContextVar(
    "spectrue_trace_phase_starts", default={}
)
_trace_events_var: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "spectrue_trace_events", default=None
)
_trace_meta_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "spectrue_trace_meta", default=None
)

# PII Regex Patterns
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b")
CC_REGEX = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _redact_text(s: str) -> str:
    if not s:
        return s

    # URL params
    # URL params (stop at &, whitespace, or end of string)
    s = re.sub(r"([?&]key=)[^&\s]+", r"\1***", s, flags=re.IGNORECASE)
    s = re.sub(r"([?&]api_key=)[^&\s]+", r"\1***", s, flags=re.IGNORECASE)
    s = re.sub(r"([?&]access_token=)[^&\s]+", r"\1***", s, flags=re.IGNORECASE)

    # Bearer tokens / auth headers
    s = re.sub(r"(Authorization:\s*Bearer\s+)[^\s]+", r"\1***", s, flags=re.IGNORECASE)
    s = re.sub(r"(Bearer\s+)[A-Za-z0-9._-]+", r"\1***", s)

    # Medical content redaction
    s = _redact_medical(s)

    # PII Redaction (Email, Phone, Credit Card)
    if _redact_pii_enabled_var.get():
        s = EMAIL_REGEX.sub("[EMAIL]", s)
        s = PHONE_REGEX.sub("[PHONE]", s)
        s = CC_REGEX.sub("[CARD]", s)

    return s


# Medical Patterns for Log Redaction
# These patterns match actionable medical instructions that should not be stored in logs.
# We preserve sha256 + safe_head for debugging while removing doses/instructions.
_MEDICAL_PATTERNS = [
    # Dosing with units (Ukrainian + English)
    r"\d+[\s,]*(?:мг|мл|г|грам|mg|ml|g|gram|mcg|мкг|iu|од|одиниц)(?:\s*/\s*(?:кг|kg|день|day|добу))?\b",
    # Intake instructions (Ukrainian)
    r"(?:приймати|пити|вживати|вводити|застосовувати|наносити|полоскати)\s+(?:по\s+)?\d+",
    # Intake instructions (English)
    r"(?:take|consume|administer|apply|inject|drink)\s+\d+",
    # Frequency patterns (Ukrainian)
    r"\d+\s*(?:раз|рази|разів)\s*(?:на|в|per)\s*(?:день|добу|тиждень|годину)",
    # Frequency patterns (English)
    r"\d+\s*(?:times?)\s*(?:per|a|daily|weekly)\s*(?:day|hour|week)?",
    # Procedural steps with numbers
    r"(?:крок|етап|step)\s*\d+\s*[:\.]\s*[^\n]{10,50}",
    # Concentration patterns
    r"\d+[\s,]*%\s*(?:розчин|solution|концентрація|concentration)",
    # Duration patterns
    r"(?:протягом|впродовж|for|during)\s+\d+\s*(?:днів|дні|хвилин|годин|days|hours|minutes|weeks)",
]


def _redact_medical(s: str) -> str:
    """
    Redact medical dosing and instructions from text.
    
    Removes actionable medical content while preserving structure for debugging.
    Patterns are designed to catch common dosing formats without false positives.
    
    Args:
        s: Input text
        
    Returns:
        Text with medical instructions replaced by [REDACTED_MEDICAL]
    """
    if not s:
        return s

    for pattern in _MEDICAL_PATTERNS:
        s = re.sub(pattern, "[REDACTED_MEDICAL]", s, flags=re.IGNORECASE)

    return s



def _is_secret_hint(key_hint: str | None) -> bool:
    if not key_hint:
        return False
    k = key_hint.lower()
    # Explicitly check for high-entropy secrets. 
    # We do NOT match generic "key" to avoid false positives on non-secret keys (topic_key, etc).
    return any(x in k for x in ("password", "secret", "token", "api_key", "apikey", "access_key", "accesskey", "authorization", "bearer"))


def _sanitize(
    obj: Any,
    *,
    max_str: int | None = None,
    max_list: int = 100,
    max_dict: int = 200,
    key_hint: str | None = None,
) -> Any:
    # Early redaction based on key hint to prevent recursion/hashing of secrets
    if _is_secret_hint(key_hint):
        return {
            "redacted": True,
            "key_hint": key_hint,
        }

    max_override = _trace_max_override_var.get()
    if max_override is not None:
        max_str = max_override
    if max_str is None:
        max_str = 4000
    if obj is None:
        return None
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        s = _redact_text(obj)

        # Safe Payloads Logic
        safe_mode = _trace_safe_payloads_var.get()
        if safe_mode:
            # Check if this is a sensitive (large text) key for truncation purposes
            is_sensitive_text = False
            if key_hint:
                k = key_hint.lower()
                # Broad match for sensitive text keys (prompts, content, etc.)
                if any(x in k for x in ("input_text", "response_text", "article", "content", "text", "prompt", "raw_html", "snippet")):
                    is_sensitive_text = True
                
            # Determine limit
            limit = _trace_max_head_var.get() if is_sensitive_text else _trace_max_inline_var.get()

            if len(s) > limit:
                # Safe mode:
                # - Sensitive keys (prompt/content/text/etc): head-only + sha256 (no tail)
                # - Non-sensitive keys: keep head+tail + sha256 for debugging
                
                out = {
                    "len": len(s),
                    "sha256": hashlib.sha256(s.encode("utf-8")).hexdigest(),
                    "head": s[:limit],
                }
                if not is_sensitive_text:
                    out["tail"] = s[-limit:] if limit else ""
                return out
            return s

        # Legacy/Unsafe Mode (keep existing behavior for backward compat if flag is off)
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
            k_lower = key.lower()
            if k_lower in ("authorization", "api_key", "key", "openai_api_key", "tavily_api_key"):
                out[key] = "***"
            else:
                # Pass key_hint down
                out[key] = _sanitize(v, max_str=max_str, max_list=max_list, max_dict=max_dict, key_hint=key)
        return out
    return _sanitize(str(obj), max_str=max_str, max_list=max_list, max_dict=max_dict, key_hint=key_hint)


def _trace_dir() -> Path:
    # Traces in Spectrue repository root /trace directory
    # Engine is at SpectrueBack/spectrue-engine, so go up 2 levels
    p = Path(__file__).resolve().parent.parent.parent.parent.parent / "trace"
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
    def _emit_trace_id() -> bool:
        """Whether to include trace_id field in each JSONL record.

        By default we can omit it because the trace file name already contains
        the trace id (and we write one file per trace).
        """
        try:
            runtime = EngineRuntimeConfig.load_from_env()
            flag = getattr(getattr(runtime, "features", None), "trace_emit_trace_id", None)
            if flag is not None:
                return bool(flag)
        except Exception:
            pass
        env = os.getenv("SPECTRUE_TRACE_EMIT_TRACE_ID")
        if env is None:
            return False
        return env.strip().lower() in ("1", "true", "yes", "y", "on")

    @staticmethod
    def start(trace_id: str, *, runtime: EngineRuntimeConfig | None = None) -> TraceContext:
        # Centralized config: trace is local-only and can be disabled via runtime.features.trace_enabled.
        runtime = runtime or EngineRuntimeConfig.load_from_env()
        enabled = bool(is_local_run() and runtime.features.trace_enabled)
        _trace_id_var.set(trace_id)
        _trace_enabled_var.set(enabled)
        _redact_pii_enabled_var.set(runtime.features.log_redaction)
        _trace_safe_payloads_var.set(runtime.features.trace_safe_payloads)
        _trace_max_head_var.set(runtime.debug.trace_max_head_chars)
        _trace_max_inline_var.set(runtime.debug.trace_max_inline_chars)
        if enabled:
            _trace_events_var.set([])
            _trace_meta_var.set(
                {
                    "trace_id": trace_id,
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            # Clean up old traces, keep only the latest one
            try:
                trace_dir = _trace_dir()
                for pattern in ("*.json", "*.jsonl"):
                    for old_file in trace_dir.glob(pattern):
                        try:
                            old_file.unlink()
                        except Exception:
                            pass
            except Exception:
                pass

            payload = {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if Trace._emit_trace_id():
                payload["trace_id"] = trace_id
            Trace.event("trace.start", payload)
        return TraceContext(trace_id=trace_id, enabled=enabled)

    @staticmethod
    def stop() -> None:
        tid = current_trace_id()
        if trace_enabled() and tid:
            payload = {}
            if Trace._emit_trace_id():
                payload["trace_id"] = tid
            Trace.event("trace.stop", payload)
            events = _trace_events_var.get() or []
            meta = _trace_meta_var.get() or {}
            meta.setdefault("trace_id", tid)
            meta.setdefault("started_at", time.strftime("%Y-%m-%d %H:%M:%S"))
            meta["stopped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                safe_tid = "".join(c if c.isalnum() or c in "._-" else "_" for c in tid)
                path = _trace_dir() / f"{safe_tid}.json"
                with path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "trace_id": meta.get("trace_id"),
                            "started_at": meta.get("started_at"),
                            "stopped_at": meta.get("stopped_at"),
                            "events": events,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                pass
        _trace_enabled_var.set(False)
        _trace_id_var.set(None)
        _trace_events_var.set(None)
        _trace_meta_var.set(None)

    @staticmethod
    def event(name: str, data: Any | None = None) -> None:
        if not trace_enabled():
            return
        tid = current_trace_id()
        if not tid:
            return
        events = _trace_events_var.get()
        if events is None:
            return

        rec = {
            "ts_ms": _now_ms(),
            "event": str(name),
            "data": _sanitize(data),
        }
        if Trace._emit_trace_id():
            rec["trace_id"] = tid
        try:
            events.append(rec)
        except Exception:
            # Tracing must never break the main flow.
            return

    @staticmethod
    def event_full(name: str, data: Any | None = None) -> None:
        """
        Emit event without safe-payload truncation (still redacts secrets/PII).
        Intended for debugging prompts/contracts; use sparingly.
        """
        prev = _trace_safe_payloads_var.get()
        prev_override = _trace_max_override_var.get()
        try:
            _trace_safe_payloads_var.set(False)
            _trace_max_override_var.set(250000)  # allow large payloads for debugging
            Trace.event(name, data)
        finally:
            _trace_max_override_var.set(prev_override)
            _trace_safe_payloads_var.set(prev)

    @staticmethod
    def progress_cost_delta(*, stage: str, delta: int, total: int) -> None:
        Trace.event(
            "progress.cost_delta",
            {
                "stage": stage,
                "delta": int(delta),
                "total": int(total),
            },
        )

    @staticmethod
    def phase_start(phase: str, *, meta: dict[str, Any] | None = None) -> None:
        if not trace_enabled():
            return
        starts = dict(_trace_phase_starts.get() or {})
        starts[phase] = _now_ms()
        _trace_phase_starts.set(starts)
        payload = {"phase": phase}
        if meta:
            payload["meta"] = meta
        Trace.event("phase.start", payload)

    @staticmethod
    def phase_end(phase: str, *, meta: dict[str, Any] | None = None) -> int | None:
        if not trace_enabled():
            return None
        starts = dict(_trace_phase_starts.get() or {})
        start_ms = starts.pop(phase, None)
        _trace_phase_starts.set(starts)
        duration_ms = _now_ms() - start_ms if start_ms is not None else None
        payload = {"phase": phase, "duration_ms": duration_ms}
        if meta:
            payload["meta"] = meta
        Trace.event("phase.end", payload)
        return duration_ms

    @staticmethod
    def reason_code(
        *,
        code: str,
        phase: str,
        action: str,
        label: str | None = None,
        claim_id: str | None = None,
    ) -> None:
        payload = {
            "code": code,
            "phase": phase,
            "action": action,
            "label": label,
        }
        if claim_id:
            payload["claim_id"] = claim_id
        Trace.event("reason_code", payload)

    @staticmethod
    @contextmanager
    def phase_span(phase: str, *, meta: dict[str, Any] | None = None):
        Trace.phase_start(phase, meta=meta)
        try:
            yield
        finally:
            Trace.phase_end(phase, meta=meta)
