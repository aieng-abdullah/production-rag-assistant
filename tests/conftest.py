"""Shared test fixtures.

The Voyage pacer (`src/voyage_pacer`) sleeps for real to respect the
account's 3 RPM / 10K TPM budget. Under test that is pure latency: a
suite that paces itself would take minutes and still prove nothing,
because every Voyage call is mocked.

So the pacer is opened up for every test by default. Tests that
specifically exercise pacing set their own limits — and any that need to
observe a real sleep patch `sleep` and assert on the call.
"""

import os

# The suite must pass with no secrets configured. GitHub does not expose
# repository secrets to fork pull requests, so this test:
#
#   test_api_documents.py::test_startup_marks_stale_processing_as_failed
#
# is the only one that enters the FastAPI lifespan, and the only one that
# calls Config.validate(). Without this it raised "No LLM provider API key
# found" on every fork PR and blocked the contributor, while passing
# locally purely because a developer's .env happened to be present.
#
# `Config` reads env at import time, so the default has to be set before the
# `src.config` import below — not in a fixture, which runs too late.
#
# GROQ only, deliberately. tests/test_config_validate.py asserts that
# clearing GROQ alone makes validate() raise, and that only holds while
# ANTHROPIC_API_KEY and OPENAI_API_KEY stay empty.
#
# Assignment, not setdefault: ci.yml passes the key through as
# `GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}`, and on a fork that secret
# resolves to an *empty string* rather than being absent. os.environ.setdefault
# only fills a missing key, so it left `GROQ_API_KEY=""` in place and the
# lifespan still raised. Testing for truthiness covers both cases: unset,
# and set-but-empty.
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "test-key-not-used"

import pytest  # noqa: E402  (must follow the env default above)

from src import voyage_pacer  # noqa: E402
from src.config import Config  # noqa: E402


@pytest.fixture(autouse=True)
def no_voyage_pacing(monkeypatch):
    """Disable token pacing unless a test opts back in.

    Also clears the rolling window between tests, since it is
    process-wide: leftover reservations from one test would otherwise
    make the next one wait for a budget it never spent.
    """
    monkeypatch.setattr(Config, "VOYAGE_TPM_LIMIT", 10_000_000)
    monkeypatch.setattr(Config, "VOYAGE_RPM_LIMIT", 1_000_000)
    monkeypatch.setattr(Config, "VOYAGE_MAX_WAIT_S", 0.0)
    voyage_pacer.reset()
    yield
    voyage_pacer.reset()
