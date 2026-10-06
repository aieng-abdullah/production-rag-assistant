"""Boilerplate filter keeps prompt slots for real document content."""

from src.retrieval.boilerplate import drop_boilerplate, is_boilerplate

RG_HEADER = (
    "See discussions, stats, and author profiles for this publication at: "
    "https://www.researchgate.net/publication/314116018\n"
    "Improving Machine Learning Ability with Fine-Tuning\n"
    "Article · February 2017\nDOI: 10.48550/arXiv.1702.08563\n"
    "CITATIONS\n12\nREADS\n6,267"
)
RG_PROFILE = (
    "John Lalor\nUniversity of Notre Dame\n"
    "51 PUBLICATIONS  577 CITATIONS\nSEE PROFILE"
)
RG_UPLOAD_NOTICE = (
    "All content following this page was uploaded by John Lalor on "
    "03 November 2017.\nThe user has requested enhancement of the downloaded file."
)
REAL_CONTENT = (
    "However, it is difficult to create a large dataset to train "
    "the ability of deep neural network models (DNNs)."
)


class TestIsBoilerplate:
    def test_researchgate_header_detected(self):
        assert is_boilerplate(RG_HEADER) is True

    def test_profile_nav_detected(self):
        assert is_boilerplate(RG_PROFILE) is True

    def test_upload_notice_detected(self):
        assert is_boilerplate(RG_UPLOAD_NOTICE) is True

    def test_real_content_kept(self):
        assert is_boilerplate(REAL_CONTENT) is False

    def test_empty_text_not_filtered(self):
        assert is_boilerplate("") is False
        assert is_boilerplate(None) is False

    def test_marker_substring_not_overbroad(self):
        """'citations' alone must not trigger — legal/academic prose uses it."""
        assert is_boilerplate("The citations in [SOURCE 1] support this claim.") is False


class TestDropBoilerplate:
    def test_removes_junk_keeps_order(self):
        chunks = [
            {"text": RG_HEADER, "chunk_id": "junk1"},
            {"text": REAL_CONTENT, "chunk_id": "real"},
            {"text": RG_PROFILE, "chunk_id": "junk2"},
        ]
        kept = drop_boilerplate(chunks)
        assert [c["chunk_id"] for c in kept] == ["real"]

    def test_all_boilerplate_returns_original(self):
        """Degenerate corpus: empty prompt slots are worse than furniture."""
        chunks = [{"text": RG_HEADER}, {"text": RG_PROFILE}]
        assert drop_boilerplate(chunks) == chunks

    def test_empty_list(self):
        assert drop_boilerplate([]) == []

    def test_no_boilerplate_passthrough(self):
        chunks = [{"text": REAL_CONTENT}]
        assert drop_boilerplate(chunks) == chunks
