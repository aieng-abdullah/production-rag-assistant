# Contributing to Production RAG Assistant

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue using the **Bug Report** template. Include:
- A clear description of the problem
- Steps to reproduce
- Expected behavior
- Your environment (OS, Python version, etc.)

### Suggesting Features

If you have an idea for a new feature, please open an issue using the **Feature Request** template. Include:
- A clear description of the feature
- Why it would be useful
- Any implementation ideas you have

### Submitting Changes

1. Fork the repository
2. Create a new branch (`git checkout -b type/short-slug`) — see [`AGENTS.md`](AGENTS.md)
3. Make your changes
4. Run the checks below (see [Running the checks](#running-the-checks))
5. Commit your changes (`git commit -m 'fix: short description'`)
6. Push to the branch (`git push origin type/short-slug`)
7. Open a pull request and fill in the template

## Development Setup

> **[`AGENTS.md`](AGENTS.md) is the fuller reference** — branch naming, commit format,
> architecture, and the rules this codebase follows. Read it before your first PR.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/production-rag-assistant.git
   cd production-rag-assistant
   ```

2. Install the Python dependencies (**Python 3.12+**):
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the environment template and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   At minimum `GROQ_API_KEY`, `VOYAGE_API_KEY` and `JWT_SECRET` are required. The app
   refuses to start without them, by design — see `Config.validate()` in `src/config.py`.

4. Install the frontend dependencies (only if you are touching `frontend/`):
   ```bash
   cd frontend && npm install && cd ..
   ```

## Working from a fork

**Your CI will sit at "Waiting for status to be reported" until a maintainer approves the run.**

This is not a failure, and it is not something you can fix. GitHub does not pass repository
secrets to fork pull requests, so the `test` job cannot see `GROQ_API_KEY`. Open your PR
as normal, and ping a maintainer — it will be approved.

If the checks show **red**, that is a real failure and it is worth reading. If they show
**"Expected — Waiting for status to be reported"**, they simply have not started.

## Running the checks

These are the exact commands CI runs. Run all three before you push:

```bash
ruff check src tests eval alembic
pytest tests/ -v -m "not slow" --cov-fail-under=70
cd frontend && npm run lint && npm run build
```

A plain `pytest` will not reproduce CI — it skips the lint gate and the coverage floor.

Tests that hit Voyage are marked `@pytest.mark.slow` and are excluded from the suite above.

## Which gates are enforced where

| Gate | Where it runs | Notes |
|---|---|---|
| `lint` | CI, on every PR and on `main` | `ruff check` + frontend lint |
| `test` | CI, on every PR and on `main` | fast tests only, no coverage gate in CI |
| coverage ≥ 70% | **local only** | CI passes `--cov-fail-under=0`; keep it green locally |
| citation gates (`verify_eval.py`) | **local only** | free tier, no API cost |
| retrieval gate (`retrieval_precision.py`) | **local only** | free tier, but slow — roughly 13 minutes |
| Ragas metrics (`eval_runner.py`) | **local only** | needs funded Groq + Voyage keys; cannot finish on free tiers |

`make eval` runs the two local-only eval gates. See [`AGENTS.md`](AGENTS.md) for why they
are not in CI.

## Pull request expectations

- Branch names are `type/short-slug` — `feat/`, `fix/`, `chore/`, `docs/`, `test/`, `ci/`, `refactor/`, `security/`.
- Commits follow Conventional Commits (`type: subject`, lowercase, ≤50 characters).
- Fill in the pull request template. If your change needs a risk note, include one.
- Keep diffs reviewable — roughly 400 lines. Split larger work across branches.
- Read your own diff before you push. It is the cheapest review you will ever get.

## Reporting a security issue

Please do not open a public issue. Email the maintainer privately instead.

## Code of conduct

Be decent to each other. Reviews here are about the code, never the person who wrote it.