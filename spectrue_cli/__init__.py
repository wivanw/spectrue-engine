# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""
Spectrue CLI Module

Provides command-line tools for pipeline management and verification.

Commands:
- pipeline list: List available pipeline profiles
- pipeline validate <name>: Validate a pipeline profile
- pipeline graph <name>: Export pipeline graph (Mermaid)
- pipeline run <claim_file> --profile <name>: Run verification

Usage:
    python -m spectrue_cli pipeline list
    python -m spectrue_cli pipeline validate normal
    python -m spectrue_cli pipeline run claims.json --profile deep
"""

from spectrue_cli.pipeline_cmd import main

__all__ = ["main"]

if __name__ == "__main__":
    main()

