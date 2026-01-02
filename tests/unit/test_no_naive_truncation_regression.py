# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from pathlib import Path

FORBIDDEN = (":4000", ":8000")
ROOT = Path(__file__).resolve().parents[2]
TARGET_FILES = [
    ROOT / "spectrue_core" / "verification" / "pipeline" / "pipeline.py",
    ROOT / "spectrue_core" / "agents" / "skills" / "claims.py",
]


def test_no_naive_truncation_regression():
    for file_path in TARGET_FILES:
        content = file_path.read_text()
        for pattern in FORBIDDEN:
            assert pattern not in content, f"Found forbidden slice pattern {pattern} in {file_path}"
