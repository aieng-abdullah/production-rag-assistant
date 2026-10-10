"""Shared test fixtures.

The Voyage pacer (`src/voyage_pacer`) sleeps for real to respect the
account's 3 RPM / 10K TPM budget. Under test that is pure latency: a
suite that paces itself would take minutes and still prove nothing,
because every Voyage call is mocked.

So the pacer is opened up for every test by default. Tests that
specifically exercise pacing set their own limits — and any that need to
observe a real sleep patch `sleep` and assert on the call.
"""

import pytest

from src import voyage_pacer
from src.config import Config


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
