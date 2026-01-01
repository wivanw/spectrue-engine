# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Any

from spectrue_core.constants import DEFAULT_RELATIVE_WINDOW_DAYS, DEFAULT_TIME_GRANULARITY_DAYS
from spectrue_core.schema.signals import TimeGranularity, TimeWindow


def _parse_iso_date(raw: Any) -> date | None:
    if not raw:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, datetime):
        return raw.date()
    try:
        value = str(raw).strip()
        if not value:
            return None
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value).date()
    except Exception:
        return None


def _coerce_granularity(raw: Any) -> TimeGranularity | None:
    if not raw:
        return None
    try:
        return TimeGranularity(str(raw).strip().lower())
    except ValueError:
        return None


def _select_best_signal(time_signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not time_signals:
        return None
    best = None
    best_conf = -1.0
    for signal in time_signals:
        if not isinstance(signal, dict):
            continue
        conf = signal.get("confidence")
        if isinstance(conf, (int, float)) and conf > best_conf:
            best = signal
            best_conf = float(conf)
        if best is None:
            best = signal
    return best


def normalize_time_window(
    time_signals: list[dict[str, Any]] | None,
    *,
    reference_date: date | None = None,
    default_relative_days: int = DEFAULT_RELATIVE_WINDOW_DAYS,
) -> TimeWindow:
    """
    Normalize LLM-extracted temporal anchors into a conservative time window.
    """
    if reference_date is None:
        reference_date = date.today()

    signal = _select_best_signal(time_signals or [])
    if not signal:
        return TimeWindow(
            start_date=None,
            end_date=None,
            granularity=None,
            source_signal="missing",
            confidence=0.0,
        )

    granularity = _coerce_granularity(signal.get("granularity"))
    start = _parse_iso_date(signal.get("start_date") or signal.get("start"))
    end = _parse_iso_date(signal.get("end_date") or signal.get("end"))

    if not start and not end:
        anchor = _parse_iso_date(signal.get("date") or signal.get("anchor_date"))
        if anchor:
            start = anchor
            end = anchor
            if granularity is None:
                granularity = TimeGranularity.DAY

    if not start and not end:
        year = signal.get("year")
        month = signal.get("month")
        if isinstance(year, (int, str)):
            try:
                year_int = int(year)
            except Exception:
                year_int = None
        else:
            year_int = None

        if isinstance(month, str) and "-" in month:
            try:
                year_part, month_part = month.split("-", 1)
                year_int = int(year_part)
                month_int = int(month_part)
            except Exception:
                month_int = None
            else:
                month = month_int

        if year_int and isinstance(month, (int, str)):
            try:
                month_int = int(month)
                last_day = calendar.monthrange(year_int, month_int)[1]
                start = date(year_int, month_int, 1)
                end = date(year_int, month_int, last_day)
                if granularity is None:
                    granularity = TimeGranularity.MONTH
            except Exception:
                start = None
                end = None
        elif year_int:
            start = date(year_int, 1, 1)
            end = date(year_int, 12, 31)
            if granularity is None:
                granularity = TimeGranularity.YEAR

    if not start and not end:
        rel_days = signal.get("relative_days") or signal.get("days")
        try:
            rel_int = int(rel_days)
        except Exception:
            rel_int = None
        if rel_int is None:
            rel_int = default_relative_days
        start = reference_date - timedelta(days=max(rel_int, 0))
        end = reference_date
        if granularity is None:
            granularity = TimeGranularity.RELATIVE

    if start and not end:
        span = DEFAULT_TIME_GRANULARITY_DAYS.get(granularity.value, 0) if granularity else 0
        end = start + timedelta(days=span)

    if end and not start:
        span = DEFAULT_TIME_GRANULARITY_DAYS.get(granularity.value, 0) if granularity else 0
        start = end - timedelta(days=span)

    if start and end and start > end:
        start, end = end, start

    source_signal = (
        signal.get("source_signal")
        or signal.get("source")
        or signal.get("anchor")
        or signal.get("text")
    )

    return TimeWindow(
        start_date=start.isoformat() if start else None,
        end_date=end.isoformat() if end else None,
        granularity=granularity,
        source_signal=str(source_signal) if source_signal else None,
        confidence=signal.get("confidence") if isinstance(signal.get("confidence"), (int, float)) else None,
    )


def label_evidence_timeliness(
    sources: list[dict],
    *,
    time_window: TimeWindow | None,
) -> list[dict]:
    """
    Label each evidence item with timeliness status.
    """
    start = _parse_iso_date(time_window.start_date) if time_window else None
    end = _parse_iso_date(time_window.end_date) if time_window else None

    for src in sources:
        if not isinstance(src, dict):
            continue
        published = _parse_iso_date(
            src.get("published_date")
            or src.get("published_at")
            or src.get("published")
        )
        if not published or not start or not end:
            src["timeliness_status"] = "unknown_date"
            continue
        if start <= published <= end:
            src["timeliness_status"] = "in_window"
        else:
            src["timeliness_status"] = "outdated"
    return sources

