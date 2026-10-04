"""Provenance metadata (PLAN PR-4b-iv): pinpoint, hashes, doc inputs."""

import hashlib
from unittest.mock import patch

from src.generation.Citation_system import build_source
from src.generation.chain import _build_sources
from src.ingestion.chunker import chunk_pages
from src.ingestion.pipeline import ingest

CHUNK = {
    "text": "Whoever desires any Court to give judgment must prove it. Section 42 applies.",
    "doc_id": "evidence_act",
    "page_num": 3,
    "chunk_index_in_page": 2,
    "filename": "indian_evidence_act_1872.pdf",
    "content_hash": "abc123",
    "doc_date": "1872-01-01",
    "doc_version": "revised-1891",
    "jurisdiction": "India",
    "doc_hash": "d" * 64,
    "ingested_at": "2026-10-04T00:00:00+00:00",
}


def test_source_numbering_is_stable_and_one_based():
    chunks = [{"doc_id": "d", "page_num": 1, "text": "x"}] * 3
    sources = _build_sources(chunks)
    assert [s.source_id for s in sources] == [1, 2, 3]


def test_pinpoint_page_paragraph_section():
    source = build_source(CHUNK, source_id=5)

    assert source.source_id == 5
    assert source.pinpoint == {"page": 3, "paragraph": 2, "section": "Section 42"}
    assert source.content_hash == "abc123"
    assert source.provenance["doc_date"] == "1872-01-01"
    assert source.provenance["jurisdiction"] == "India"
    assert source.provenance["doc_hash"] == "d" * 64


def test_pinpoint_section_absent_is_none():
    source = build_source({"text": "no numbering here " * 10}, source_id=1)
    assert source.pinpoint["section"] is None


def test_pinpoint_section_complex_numbering():
    text = "burden of proof Section 112A(2) of the Act " * 10
    source = build_source({"text": text}, source_id=1)
    assert source.pinpoint["section"] == "Section 112A(2)"


def test_pinpoint_missing_metadata_defaults():
    source = build_source({"text": "hello world " * 10}, source_id=9)

    assert source.pinpoint["page"] == -1
    assert source.pinpoint["paragraph"] == -1
    assert source.content_hash == ""
    assert source.provenance["doc_date"] == ""


def test_chunker_content_hash_is_sha256_of_text():
    text = "Some chunk body text that is long enough for the chunker. " * 3
    chunks = chunk_pages([{"text": text, "page_num": 1, "doc_id": "d"}])

    assert chunks
    expected = hashlib.sha256(chunks[0]["text"].encode("utf-8")).hexdigest()
    assert chunks[0]["content_hash"] == expected


class TestIngestProvenanceStamping:
    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_provenance_inputs_stamped(self, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [{"text": "A" * 200, "page_num": 1, "doc_id": "d1"}]
        mock_embed.side_effect = lambda chunks: chunks
        mock_upsert.return_value = 1

        ingest(
            "/fake/doc.pdf",
            provenance={"date": "1872", "version": "v1", "jurisdiction": "India"},
        )

        stamped = mock_upsert.call_args[0][0]
        assert stamped[0]["doc_date"] == "1872"
        assert stamped[0]["doc_version"] == "v1"
        assert stamped[0]["jurisdiction"] == "India"
        assert stamped[0]["ingested_at"]
        assert stamped[0]["doc_hash"] == ""

    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_provenance_defaults_empty(self, mock_extract, mock_embed, mock_upsert):
        mock_extract.return_value = [{"text": "A" * 200, "page_num": 1, "doc_id": "d1"}]
        mock_embed.side_effect = lambda chunks: chunks
        mock_upsert.return_value = 1

        ingest("/fake/doc.pdf")

        stamped = mock_upsert.call_args[0][0]
        assert stamped[0]["doc_date"] == ""
        assert stamped[0]["jurisdiction"] == ""

    @patch("src.ingestion.pipeline.upsert_chunks")
    @patch("src.ingestion.pipeline.embed_chunks")
    @patch("src.ingestion.pipeline.extract_pages")
    def test_doc_hash_sha256_of_real_file(
        self, mock_extract, mock_embed, mock_upsert, tmp_path
    ):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake bytes for hashing")
        mock_extract.return_value = [{"text": "A" * 200, "page_num": 1, "doc_id": "d1"}]
        mock_embed.side_effect = lambda chunks: chunks
        mock_upsert.return_value = 1

        ingest(str(pdf))

        stamped = mock_upsert.call_args[0][0]
        expected = hashlib.sha256(b"%PDF-1.4 fake bytes for hashing").hexdigest()
        assert stamped[0]["doc_hash"] == expected
