#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v lint-imports >/dev/null 2>&1; then
  echo "lint-imports not found. Install it with: pip install import-linter" >&2
  exit 1
fi

lint-imports --config "$ROOT_DIR/.importlinter"
