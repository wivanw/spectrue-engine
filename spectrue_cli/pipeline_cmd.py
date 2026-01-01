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
"""
Pipeline CLI Commands

Provides CLI commands for pipeline profile management.

Commands:
- list: List available profiles
- validate: Validate a profile
- graph: Export pipeline graph (Mermaid)
- run: Run verification with a profile (future)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn


def cmd_list(args: argparse.Namespace) -> int:
    """List available pipeline profiles."""
    from spectrue_core.pipeline_builder import list_profiles, PROFILES_DIR

    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else PROFILES_DIR

    profiles = list_profiles(profiles_dir)

    if not profiles:
        print(f"No profiles found in {profiles_dir}", file=sys.stderr)
        return 1

    print(f"Available pipeline profiles ({profiles_dir}):")
    for name in sorted(profiles):
        print(f"  - {name}")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a pipeline profile."""
    from spectrue_core.pipeline_builder import load_profile, ValidationError, PROFILES_DIR

    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else PROFILES_DIR
    profile_name = args.name

    try:
        profile = load_profile(profile_name, profiles_dir=profiles_dir, validate=True)
        print(f"✓ Profile '{profile_name}' is valid")
        print(f"  Version: {profile.version}")
        print(f"  Description: {profile.description}")
        print(f"  Budget class: {profile.budget.budget_class.value}")
        print(f"  Max credits: {profile.budget.max_credits}")
        print(f"  Phases: {', '.join(profile.phases.enabled)}")
        return 0

    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    except ValidationError as e:
        print(f"✗ Profile '{profile_name}' validation failed:", file=sys.stderr)
        for error in e.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"✗ Error loading profile '{profile_name}': {e}", file=sys.stderr)
        return 1


def cmd_graph(args: argparse.Namespace) -> int:
    """Export pipeline profile as Mermaid graph."""
    from spectrue_core.pipeline_builder import load_profile, ValidationError, PROFILES_DIR

    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else PROFILES_DIR
    profile_name = args.name

    try:
        profile = load_profile(profile_name, profiles_dir=profiles_dir, validate=True)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    except ValidationError as e:
        print(f"✗ Profile validation failed: {e}", file=sys.stderr)
        return 1

    # Generate Mermaid flowchart
    lines = [
        "```mermaid",
        "flowchart TD",
        f"    subgraph {profile.name}[{profile.name.upper()} Profile]",
        "    direction TB",
    ]

    # Add phases as nodes
    phases = list(profile.phases.enabled)
    for i, phase_id in enumerate(phases):
        node_id = f"P{phase_id.replace('-', '_')}"
        label = f"Phase {phase_id}"
        lines.append(f"    {node_id}[{label}]")

    # Add edges between phases
    for i in range(len(phases) - 1):
        curr = f"P{phases[i].replace('-', '_')}"
        next_p = f"P{phases[i + 1].replace('-', '_')}"
        lines.append(f"    {curr} --> {next_p}")

    # Add stop decision for deep profiles
    if profile.stop_policy.enabled:
        lines.append("    STOP{{Stop Decision}}")
        if phases:
            last_phase = f"P{phases[-1].replace('-', '_')}"
            lines.append(f"    {last_phase} -.-> STOP")

    lines.append("    end")

    # Add metadata
    lines.append("")
    lines.append(f"    %% Budget: {profile.budget.budget_class.value} (max {profile.budget.max_credits} credits)")
    lines.append(f"    %% Search: {profile.search.depth}, max {profile.search.max_results} results")
    if profile.fulltext.enabled:
        lines.append(f"    %% Fulltext: top {profile.fulltext.topk}")

    lines.append("```")

    output = "\n".join(lines)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Graph written to {args.output}")
    else:
        print(output)

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Build and (optionally) execute a plan using a pipeline profile."""
    from spectrue_core.pipeline_builder import (
        PipelineBuilder,
        load_profile,
        PROFILES_DIR,
        ValidationError,
    )

    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else PROFILES_DIR

    # Load claims JSON
    claims_path = Path(args.claims_file)
    if not claims_path.exists():
        print(f"✗ Claims file not found: {claims_path}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(claims_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"✗ Failed to parse JSON: {e}", file=sys.stderr)
        return 1

    if isinstance(payload, dict) and "claims" in payload:
        claims = payload.get("claims")
    else:
        claims = payload

    if not isinstance(claims, list):
        print("✗ Claims JSON must be a list or {claims:[...]}.", file=sys.stderr)
        return 1

    # Parse overrides from CLI flags
    overrides: dict = {}
    if args.budget is not None:
        overrides.setdefault("budget", {})["max_credits"] = int(args.budget)
    if args.max_results is not None:
        overrides.setdefault("search", {})["max_results"] = int(args.max_results)
    if args.fulltext_topk is not None:
        overrides.setdefault("fulltext", {})["topk"] = int(args.fulltext_topk)

    try:
        profile = load_profile(args.profile, profiles_dir=profiles_dir)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    except ValidationError as e:
        print(f"✗ Profile '{args.profile}' validation failed:", file=sys.stderr)
        for err in e.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    builder = PipelineBuilder(profile, overrides=overrides or None)
    plan = builder.build_plan(claims)

    # Output
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Plan JSON written to {out_path}")
    else:
        print(plan.summary())

    if args.dry_run:
        print("\n(dry-run) Plan built successfully. Execution is not implemented in CLI yet.")
        return 0

    print(
        "\nExecution is not implemented in this CLI command yet. "
        "Use the main Spectrue engine / API to execute checks.",
        file=sys.stderr,
    )
    return 2


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="spectrue-cli pipeline",
        description="Pipeline profile management commands",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List available pipeline profiles",
    )
    list_parser.add_argument(
        "--profiles-dir",
        help="Path to profiles directory (default: pipelines/)",
    )
    list_parser.set_defaults(func=cmd_list)

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a pipeline profile",
    )
    validate_parser.add_argument(
        "name",
        help="Profile name to validate",
    )
    validate_parser.add_argument(
        "--profiles-dir",
        help="Path to profiles directory",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # graph command
    graph_parser = subparsers.add_parser(
        "graph",
        help="Export pipeline profile as Mermaid graph",
    )
    graph_parser.add_argument(
        "name",
        help="Profile name to export",
    )
    graph_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
    )
    graph_parser.add_argument(
        "--profiles-dir",
        help="Path to profiles directory",
    )
    graph_parser.set_defaults(func=cmd_graph)

    # run command (stub)
    run_parser = subparsers.add_parser(
        "run",
        help="Build an ExecutionPlan with a pipeline profile (and optionally export it)",
    )
    run_parser.add_argument(
        "claims_file",
        help="Path to JSON file with claims (list or {claims:[...]})",
    )
    run_parser.add_argument(
        "--profile", "-p",
        default="normal",
        help="Pipeline profile to use (default: normal)",
    )
    run_parser.add_argument(
        "--budget", "-b",
        type=int,
        help="Budget override (max credits)",
    )
    run_parser.add_argument(
        "--max-results",
        type=int,
        help="Search max_results override",
    )
    run_parser.add_argument(
        "--fulltext-topk",
        type=int,
        help="Fulltext topk override",
    )
    run_parser.add_argument(
        "--output-json",
        help="Write execution plan JSON to this path",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build + print/export plan (default)",
    )
    run_parser.add_argument(
        "--profiles-dir",
        help="Path to profiles directory",
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> NoReturn:
    """Main entry point for pipeline CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        exit_code = args.func(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

