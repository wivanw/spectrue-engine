# Contributing to Spectrue Engine

Thank you for your interest in contributing to Spectrue Engine! We strictly follow a standardized Git Flow process to ensure code quality and stability.

## 🚀 How to Contribute Code (The Right Way)

You **must** use a new branch and a Pull Request (PR) for all changes. Direct commits to `main` are not allowed.

### Step 1. Create a New Branch (Locally)
In your terminal (inside the project folder):

```bash
# Create and switch to a new feature branch
git checkout -b feature/initial-setup
```

**Branch Naming Convention:**
- `feature/name` - for new features
- `fix/name` - for bug fixes
- `refactor/name` - for code restructuring
- `docs/name` - for documentation updates

### Step 2. Push Your Branch to GitHub

```bash
# Push the branch and set upstream tracking
git push -u origin feature/initial-setup
```

### Step 3. Open a Pull Request (PR)
1. Go to the [GitHub repository](https://github.com/wivanw/spectrue-engine).
2. You will see a banner "Compare & pull request". Click it.
3. Fill in the PR template: describe **what** you changed and **why**.
4. Request a review from maintainers.

---

## 🛠️ Development Guidelines

1. **Linting:** Run `ruff check .` before committing.
2. **Testing:** Run `pytest` to ensure no regressions.
3. **Type Checking:** Run `mypy .` for type safety.

## 📝 License

By contributing, you agree that your contributions will be licensed under the **AGPLv3**.
