# Contributing to Spectrue Engine

First off, thank you for considering contributing to Spectrue Engine! It's people like you that make Spectrue such a great tool.

## 🚀 The Golden Rule: Git Flow & Pull Requests

To maintain code quality and stability, we follow a strict process:
**Direct commits to `main` are NOT allowed.** You must use a new branch and a Pull Request.

### How to Contribute (Step-by-Step)

**1. Create a New Branch**  
Always create a new branch for your work. Do not work on `main`.

```bash
# Good examples:
git checkout -b feature/search-optimization
git checkout -b fix/api-timeout
git checkout -b docs/update-readme
```

**Branch Naming Convention:**
- `feature/name` - for new features
- `fix/name` - for bug fixes
- `refactor/name` - for code restructuring
- `docs/name` - for documentation updates

**2. Commit Your Changes**  
Write clear, concise commit messages (Conventional Commits preferred).

```bash
git commit -m "feat: implement vector verification logic"
```

**3. Push to GitHub**  
Push your branch to the remote repository.

```bash
git push -u origin feature/search-optimization
```

**4. Create a Pull Request (PR)**
- Go to the repository on GitHub.
- Click "Compare & pull request".
- Describe **what** you changed and **why**.
- Request a review.

---

## 1. Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
1. Fork the repository on GitHub (if you don't have write access).
2. Clone the repository locally:
   ```bash
   git clone https://github.com/wivanw/spectrue-engine.git
   cd spectrue-engine
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
4. Install dependencies in editable mode with dev tools:
   ```bash
   pip install -e ".[dev]"
   ```

## 2. Code Style & Quality

We enforce high standards for code quality.

- **Linting:** We use [Ruff](https://github.com/astral-sh/ruff).
  ```bash
  ruff check .
  ```
- **Formatting:**
  ```bash
  ruff format .
  ```
- **Type Checking:** We use `mypy`.
  ```bash
  mypy .
  ```

## 3. Running Tests

Ensure all tests pass before submitting your PR.

```bash
pytest tests/unit tests/test_*.py \
  tests/integration/test_m80_orchestration.py \
  tests/integration/test_m81_calibration.py \
  tests/integration/test_verification_pipeline.py
```

## 4. Reporting Bugs

Please use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md) to report bugs. Include as much detail as possible (logs, reproduction steps).

## 5. License

By contributing, you agree that your contributions will be licensed under the **GNU Affero General Public License v3 (AGPLv3)**.

## 6. License Headers

All source files must contain the following copyright header:

```python
# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
```
