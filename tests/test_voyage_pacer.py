"""Voyage pacer tests.

The window logic is the whole point of the module, so these drive it
directly with a patched clock rather than through HTTP mocks.
"""

import pytest

from src import voyage_pacer
from src.voyage_pacer import estimate_documents_tokens, estimate_tokens


@pytest.fixture(autouse=True)
def no_real_sleeps(monkeypatch):
    """Capture pacing waits instead of serving them.

    Tests assert that a wait *would* happen. Left real, the RPM and TPM
    cases sleep for their full cap — a suite that takes two minutes to
    prove arithmetic. `monotonic` is frozen alongside `sleep` so the
    rolling window stays consistent with the recorded waits.
    """
    waits: list[float] = []
    clock = {"now": 1000.0}
    monkeypatch.setattr(voyage_pacer, "sleep", waits.append)
    monkeypatch.setattr(
        voyage_pacer, "monotonic", lambda: clock["now"] + sum(waits)
    )
    monkeypatch.setattr(voyage_pacer.Config, "VOYAGE_TPM_LIMIT", 1000)
    monkeypatch.setattr(voyage_pacer.Config, "VOYAGE_RPM_LIMIT", 1000)
    monkeypatch.setattr(voyage_pacer.Config, "VOYAGE_MAX_WAIT_S", 60.0)
    voyage_pacer.reset()
    yield waits
    voyage_pacer.reset()


@pytest.fixture
def sleeps(no_real_sleeps):
    return no_real_sleeps


class TestEstimates:
    def test_scales_with_length(self):
        assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40)

    def test_empty_string_still_costs_something(self):
        assert estimate_tokens("") >= 1

    def test_documents_sum(self):
        assert estimate_documents_tokens(["x" * 40, "y" * 80]) == (
            estimate_tokens("x" * 40) + estimate_tokens("y" * 80)
        )

    def test_empty_batch_is_free(self):
        assert estimate_documents_tokens([]) == 0


class TestWindow:
    def test_first_call_never_waits(self):
        assert voyage_pacer.reserve(10) == 0.0

    def test_reservations_accumulate(self):
        voyage_pacer.reserve(10)
        voyage_pacer.reserve(10)
        assert voyage_pacer.snapshot(voyage_pacer.EMBEDDINGS) == {"tokens": 20, "calls": 2}

    def test_rpm_limit_forces_a_wait(self, monkeypatch):
        monkeypatch.setattr(voyage_pacer.Config, "VOYAGE_RPM_LIMIT", 2)
        voyage_pacer.reserve(1)
        voyage_pacer.reserve(1)
        assert voyage_pacer.reserve(1) > 0, "third call in the window must wait"

    def test_tpm_limit_forces_a_wait(self, monkeypatch):
        voyage_pacer.reserve(900)
        assert voyage_pacer.reserve(900) > 0, "900 + 900 exceeds a 1000 budget"

    def test_wait_is_capped(self, monkeypatch):
        """One fat request must not stall a query for a minute silently."""
        monkeypatch.setattr(voyage_pacer.Config, "VOYAGE_MAX_WAIT_S", 5.0)
        voyage_pacer.reserve(990)
        assert voyage_pacer.reserve(990) <= 5.0

    def test_reset_clears_the_window(self):
        voyage_pacer.reserve(500)
        voyage_pacer.reset()
        assert voyage_pacer.snapshot(voyage_pacer.EMBEDDINGS) == {"tokens": 0, "calls": 0}


class TestObserve:
    def test_zero_actual_is_ignored(self):
        voyage_pacer.reserve(100)
        before = voyage_pacer.snapshot(voyage_pacer.EMBEDDINGS)["tokens"]
        voyage_pacer.observe(0, 100)
        assert voyage_pacer.snapshot(voyage_pacer.EMBEDDINGS)["tokens"] == before

    def test_large_drift_is_logged(self, monkeypatch):
        """A persistently wrong estimate means the chars-per-token
        constant needs revisiting, so the drift must be visible."""
        logged = []
        monkeypatch.setattr(
            voyage_pacer.logger, "debug",
            lambda message, *args: logged.append(message.format(*args)),
        )
        voyage_pacer.observe(1000, 10)
        assert logged and "estimate off by" in logged[0]

    def test_small_drift_is_not_logged(self, monkeypatch):
        logged = []
        monkeypatch.setattr(
            voyage_pacer.logger, "debug",
            lambda message, *args: logged.append(message.format(*args)),
        )
        voyage_pacer.observe(100, 95)
        assert not logged


class TestPerModelBudgets:
    """Voyage counts limits per model, so the windows must not couple.

    Measured: a 37-item eval issues 75 Voyage calls in 12.7 minutes —
    5.9 RPM — with no 429. A shared 3 RPM allowance would have failed, so
    `voyage-4-lite` and `rerank-3-lite` are counted independently.
    """

    def test_a_fat_rerank_does_not_stall_an_embedding(self):
        voyage_pacer.reserve(estimate_documents_tokens(["x" * 4000]), voyage_pacer.RERANK)
        assert voyage_pacer.reserve(10, voyage_pacer.EMBEDDINGS) == 0.0

    def test_windows_are_reported_separately(self):
        voyage_pacer.reserve(500, voyage_pacer.RERANK)
        voyage_pacer.reserve(20, voyage_pacer.EMBEDDINGS)
        snap = voyage_pacer.snapshot()
        assert snap[voyage_pacer.RERANK]["tokens"] == 500
        assert snap[voyage_pacer.EMBEDDINGS]["tokens"] == 20

    def test_rerank_still_paces_itself(self):
        voyage_pacer.reserve(900, voyage_pacer.RERANK)
        assert voyage_pacer.reserve(900, voyage_pacer.RERANK) > 0
