"""Answer-time context assembly tests (#123).

The default configuration must be a strict no-op — that is what makes
this safe to land before the widening in #125. The widening behaviour is
tested here too so #126 can tune it without re-deriving the semantics.
"""

import pytest

from src.config import Config
from src.generation.context_assembly import (
    assemble_context,
    is_widening_enabled,
)


@pytest.fixture(autouse=True)
def widening_off(monkeypatch):
    """Every test states the width it wants; default is off."""
    monkeypatch.setattr(Config, "CONTEXT_SECTION_WIDTH", 0)
    monkeypatch.setattr(Config, "CONTEXT_NEIGHBOR_WINDOW", 0)
    monkeypatch.setattr(Config, "CONTEXT_CHAR_BUDGET", 24000)


def _chunk(chunk_id, doc_id="act", section=None, text="text", index=0):
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "chunk_index": index,
    }
    if section is not None:
        chunk["section_id"] = section
    return chunk


class TestDefaultIsNoOp:
    """The guarantee this whole ticket exists to establish."""

    def test_disabled_by_default(self):
        assert is_widening_enabled() is False

    def test_returns_identical_objects_in_identical_order(self):
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        assert assemble_context(chunks, [_chunk("a"), _chunk("b"), _chunk("c")]) == chunks

    def test_needs_no_corpus(self):
        chunks = [_chunk("a")]
        assert assemble_context(chunks) == chunks

    def test_empty_retrieval_stays_empty(self):
        assert assemble_context([], [_chunk("a")]) == []


class TestSectionWidening:
    def test_hits_its_own_section_text(self):
        corpus = [
            _chunk("s1a", section="1", text="first part", index=0),
            _chunk("s1b", section="1", text="second part", index=1),
            _chunk("s2a", section="2", text="other section", index=2),
        ]
        Config.CONTEXT_SECTION_WIDTH = 1
        out = assemble_context([corpus[0]], corpus)
        assert "first part" in out[0]["text"]
        assert "other section" not in out[0]["text"]

    def test_chunks_without_section_metadata_pass_through(self):
        corpus = [_chunk("s1a", section="1", text="in section"), _chunk("legacy", text="no section")]
        Config.CONTEXT_SECTION_WIDTH = 1
        out = assemble_context([corpus[1]], corpus)
        assert out[0]["text"] == "no section"
        assert "context_widened" not in out[0]

    def test_never_merges_across_documents(self):
        corpus = [
            _chunk("a1", doc_id="act", section="1", text="act text"),
            _chunk("b1", doc_id="bill", section="1", text="bill text"),
        ]
        Config.CONTEXT_SECTION_WIDTH = 1
        out = assemble_context([corpus[0]], corpus)
        assert "bill text" not in out[0]["text"]

    def test_preserves_position_and_length(self):
        """Citation ids are positional, so order and count cannot change."""
        corpus = [_chunk(f"c{i}", section=str(i), text=f"body {i}", index=i) for i in range(6)]
        Config.CONTEXT_SECTION_WIDTH = 3
        hits = [corpus[1], corpus[4]]
        out = assemble_context(hits, corpus)
        assert len(out) == 2
        assert out[0]["chunk_id"] == "c1"
        assert out[1]["chunk_id"] == "c4"


class TestNeighborWidening:
    def test_pulls_in_adjacent_chunks(self):
        corpus = [
            _chunk("c0", text="before", index=0),
            _chunk("c1", text="target", index=1),
            _chunk("c2", text="after", index=2),
        ]
        Config.CONTEXT_NEIGHBOR_WINDOW = 1
        out = assemble_context([corpus[1]], corpus)
        assert "before" in out[0]["text"] and "after" in out[0]["text"]

    def test_never_crosses_document_boundary(self):
        corpus = [
            _chunk("a0", doc_id="act", text="act text", index=0),
            _chunk("b0", doc_id="bill", text="bill text", index=1),
        ]
        Config.CONTEXT_NEIGHBOR_WINDOW = 1
        out = assemble_context([corpus[0]], corpus)
        assert "bill text" not in out[0]["text"]

    def test_sectioned_chunk_keeps_its_text_when_section_width_is_zero(self):
        """Regression: the two knobs are independent.

        Section width 0 with neighbour width 1 used to select an empty
        section span — `max(0, (0 - 1) // 2)` is 0, so the window covered
        no chunks — and mark the chunk widened. That blanked the text of
        every section-bearing chunk *and* excluded it from the neighbour
        fallback, which is the path a chunk without section metadata
        takes. Enabling only the neighbour knob must never lose content.
        """
        corpus = [
            _chunk("s1a", section="1", text="Burden lies on the plaintiff.", index=0),
            _chunk("s1b", section="1", text="Nothing further required.", index=1),
            _chunk("s2a", section="2", text="Section 2 text.", index=2),
        ]
        Config.CONTEXT_SECTION_WIDTH = 0
        Config.CONTEXT_NEIGHBOR_WINDOW = 1
        out = assemble_context([corpus[0]], corpus)
        assert "Burden lies on the plaintiff." in out[0]["text"]
        assert "Nothing further required." in out[0]["text"]

    def test_zero_window_is_a_no_op(self):
        corpus = [_chunk("c0", text="a"), _chunk("c1", text="b")]
        Config.CONTEXT_NEIGHBOR_WINDOW = 0
        assert assemble_context([corpus[0]], corpus)[0]["text"] == "a"


class TestBudget:
    """The ceiling bounds widening. With widening off there is nothing to
    bound — retrieval already fixes the context size — so the budget is
    deliberately not applied, keeping the no-op guarantee exact."""

    def test_drops_lowest_rank_until_it_fits(self):
        Config.CONTEXT_NEIGHBOR_WINDOW = 1
        corpus = [_chunk(f"c{i}", text="x" * 100, index=i) for i in range(5)]
        Config.CONTEXT_CHAR_BUDGET = 250
        out = assemble_context([corpus[i] for i in range(5)], corpus)
        assert sum(len(c["text"]) for c in out) <= 250
        assert out[0]["chunk_id"] == "c0"  # highest rank survives

    def test_truncation_is_deterministic(self):
        Config.CONTEXT_NEIGHBOR_WINDOW = 1
        corpus = [_chunk(f"c{i}", text="y" * 50, index=i) for i in range(8)]
        Config.CONTEXT_CHAR_BUDGET = 120
        first = assemble_context(list(corpus), corpus)
        second = assemble_context(list(corpus), corpus)
        assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]

    def test_zero_budget_disables_the_ceiling(self):
        Config.CONTEXT_NEIGHBOR_WINDOW = 1
        corpus = [_chunk(f"c{i}", text="z" * 500, index=i) for i in range(4)]
        Config.CONTEXT_CHAR_BUDGET = 0
        assert len(assemble_context(list(corpus), corpus)) == 4

    def test_over_budget_logs_but_does_not_raise(self, monkeypatch):
        import src.generation.context_assembly as module

        warnings = []
        monkeypatch.setattr(module.logger, "warning", lambda msg: warnings.append(msg))
        Config.CONTEXT_NEIGHBOR_WINDOW = 1
        corpus = [_chunk(f"c{i}", text="w" * 400, index=i) for i in range(4)]
        Config.CONTEXT_CHAR_BUDGET = 500
        assemble_context(list(corpus), corpus)
        assert warnings, "exceeding the budget must be visible in the logs"


class TestPromptAndJudgeSeeTheSameText:
    """The regression this module exists to prevent.

    The quote verifier and the entailment judge both index the context by
    `citation.source_id - 1`. If either read the pre-widening list while
    the prompt read the widened one, both would still be internally
    consistent and the judge would verify text the user never saw.
    """

    @staticmethod
    def _claim(text, source_id=1, quote="quote"):
        from src.generation.schema import Citation, Claim

        return Claim(
            text=text,
            citations=[Citation(source_id=source_id, quote=quote)],
        )

    def test_widened_text_reaches_the_judge_and_the_verifier(self):
        from src.generation.schema import StructuredAnswer, verify_quotes
        from src.generation.verifier import build_batch_judge_prompt

        corpus = [
            _chunk("s1a", section="1", text="The burden of proof lies on the plaintiff.", index=0),
            _chunk("s1b", section="1", text="Nothing further is required.", index=1),
        ]
        Config.CONTEXT_SECTION_WIDTH = 1
        context = assemble_context([corpus[0]], corpus)

        quote = "The burden of proof lies on the plaintiff."
        claim = self._claim("The plaintiff bears the burden of proof.", 1, quote)

        prompt = build_batch_judge_prompt([claim], context)
        assert quote in prompt, "the judge must see the widened text"
        # The quote verifier resolves source_id 1 against the same list.
        verify_quotes(
            StructuredAnswer(
                claims=[claim], abstained=False, abstain_reason=None
            ),
            context,
        )

    def test_neighbour_only_widening_also_reaches_the_judge(self):
        from src.generation.verifier import build_batch_judge_prompt

        corpus = [
            _chunk("c0", text="Section 32 states as follows.", index=0),
            _chunk("c1", text="Burden lies on the plaintiff.", index=1),
        ]
        Config.CONTEXT_NEIGHBOR_WINDOW = 1
        context = assemble_context([corpus[1]], corpus)
        claim = self._claim("Burden lies on the plaintiff.", 1, "Burden lies on the plaintiff.")

        prompt = build_batch_judge_prompt([claim], context)
        assert "Section 32 states as follows." in prompt

    def test_disabled_widening_leaves_identical_text_for_both(self):
        from src.generation.verifier import build_batch_judge_prompt

        chunks = [_chunk("a", text="plain text")]
        context = assemble_context(chunks, [_chunk("a", text="plain text")])
        prompt = build_batch_judge_prompt([self._claim("a claim")], context)
        assert "plain text" in prompt