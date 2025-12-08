# Contributing to Spectrue Engine

First off, thank you for considering contributing to Spectrue Engine! It's people like you that make Spectrue such a great tool.

## 1. Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
1. Fork the repository on GitHub.
2. Clone your fork locally:
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

## 2. Running Tests

We use `pytest` for testing. Ensure your tests pass before submitting a PR.

```bash
pytest
```

## 3. Code Style

- We follow **PEP 8**.
- We use **Ruff** for linting and formatting.

To lint your code:
```bash
ruff check .
```

To format your code:
```bash
ruff format .
```

## 4. Submission Guidelines (Pull Requests)

1. **Fork & Branch**: Create a new branch for your feature or fix.
   ```bash
   git checkout -b feature/amazing-feature
   ```
2. **Commit**: Make your changes. Write clear, concise commit messages.
   ```bash
   git commit -m "feat: add amazing feature"
   ```
3. **Push**: Push to your fork.
   ```bash
   git push origin feature/amazing-feature
   ```
4. **Pull Request**: Open a Pull Request from your fork to the `main` branch of `spectrue-engine`.
   - Describe your changes clearly.
   - Link any relevant issues (e.g., "Fixes #123").
   - Ensure checks pass.

## 5. Reporting Bugs

Please use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md) to report bugs. Include as much detail as possible.

## 6. License

By contributing, you agree that your contributions will be licensed under the [AGPLv3 License](LICENSE).
