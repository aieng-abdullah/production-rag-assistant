"""Tests for retrieval pipeline."""
import pytest
from unittest.mock import patch, MagicMock

from src.retrieval.pipeline import retrieval
from src.retrieval.hybrid_fusion import rrf_fusion


class TestRetrievalSuccess:
    """Test successful pipeline execution with real RRF fusion."""

    @patch("src.retrieval.pipeline.rerank")
    @patch("src.retrieval.pipeline.vector_search")
    @patch("src.retrieval.pipeline.bm25_search")
    def test_pipeline_chains_steps_in_order(self, mock_bm25, mock_vector, mock_rerank):
        bm25_chunks = [
            {"chunk_id": "c1", "text": "a"},
            {"chunk_id": "c2", "text": "b"},
        ]
        vector_chunks = [
            {"chunk_id": "c1", "text": "a"},
            {"chunk_id": "c3", "text": "c"},
        ]
        mock_bm25.return_value = bm25_chunks
        mock_vector.return_value = vector_chunks
        mock_rerank.return_value = [{"chunk_id": "c1", "text": "a", "rerank_score": 0.9}]

        result = retrieval("query", MagicMock(), top_k=5)

        mock_bm25.assert_called_once()
        mock_vector.assert_called_once()
        mock_rerank.assert_called_once()
        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.9

    @patch("src.retrieval.pipeline.rerank")
    @patch("src.retrieval.pipeline.vector_search")
    @patch("src.retrieval.pipeline.bm25_search")
    def test_rrf_fusion_is_called_with_real_function(self, mock_bm25, mock_vector, mock_rerank):
        """Verify real rrf_fusion runs, not a mock."""
        bm25_chunks = [
            {"chunk_id": "c1", "text": "a"},
            {"chunk_id": "c2", "text": "b"},
        ]
        vector_chunks = [
            {"chunk_id": "c1", "text": "a"},
            {"chunk_id": "c3", "text": "c"},
        ]
        mock_bm25.return_value = bm25_chunks
        mock_vector.return_value = vector_chunks
        mock_rerank.return_value = []

        retrieval("query", MagicMock(), top_k=5)

        rrf_call = mock_rerank.call_args[0][1]
        chunk_ids = [c["chunk_id"] for c in rrf_call]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids
        assert "c3" in chunk_ids

    @patch("src.retrieval.pipeline.rerank")
    @patch("src.retrieval.pipeline.vector_search")
    @patch("src.retrieval.pipeline.bm25_search")
    def test_rrf_deduplicates_overlapping_chunks(self, mock_bm25, mock_vector, mock_rerank):
        """Real RRF fusion deduplicates chunks with the same chunk_id."""
        shared = [{"chunk_id": "c1", "text": "shared"}]
        mock_bm25.return_value = shared
        mock_vector.return_value = shared
        mock_rerank.return_value = []

        retrieval("query", MagicMock(), top_k=5)

        rrf_call = mock_rerank.call_args[0][1]
        chunk_ids = [c["chunk_id"] for c in rrf_call]
        assert chunk_ids.count("c1") == 1

    @patch("src.retrieval.pipeline.rerank")
    @patch("src.retrieval.pipeline.vector_search")
    @patch("src.retrieval.pipeline.bm25_search")
    def test_rrf_scores_are_assigned(self, mock_bm25, mock_vector, mock_rerank):
        """Real RRF fusion adds rrf_score to each chunk."""
        mock_bm25.return_value = [{"chunk_id": "c1", "text": "a"}]
        mock_vector.return_value = [{"chunk_id": "c2", "text": "b"}]
        mock_rerank.return_value = []

        retrieval("query", MagicMock(), top_k=5)

        rrf_call = mock_rerank.call_args[0][1]
        for chunk in rrf_call:
            assert "rrf_score" in chunk
            assert chunk["rrf_score"] > 0


class TestRetrievalErrors:
    @patch("src.retrieval.pipeline.bm25_search")
    def test_bm25_error_wrapped(self, mock_bm25):
        mock_bm25.side_effect = RuntimeError("bm25 failed")
        with pytest.raises(RuntimeError, match="Error while BM25 search"):
            retrieval("query", MagicMock())

    @patch("src.retrieval.pipeline.vector_search")
    @patch("src.retrieval.pipeline.bm25_search")
    def test_vector_error_wrapped(self, mock_bm25, mock_vector):
        mock_bm25.return_value = []
        mock_vector.side_effect = RuntimeError("vector failed")
        with pytest.raises(RuntimeError, match="Error while vector search"):
            retrieval("query", MagicMock())

    @patch("src.retrieval.pipeline.rrf_fusion")
    @patch("src.retrieval.pipeline.vector_search")
    @patch("src.retrieval.pipeline.bm25_search")
    def test_rrf_error_wrapped(self, mock_bm25, mock_vector, mock_rrf):
        mock_bm25.return_value = []
        mock_vector.return_value = []
        mock_rrf.side_effect = RuntimeError("rrf failed")
        with pytest.raises(RuntimeError, match="Error while RRF fusion"):
            retrieval("query", MagicMock())

    @patch("src.retrieval.pipeline.rerank")
    @patch("src.retrieval.pipeline.rrf_fusion")
    @patch("src.retrieval.pipeline.vector_search")
    @patch("src.retrieval.pipeline.bm25_search")
    def test_rerank_error_wrapped(self, mock_bm25, mock_vector, mock_rrf, mock_rerank):
        mock_bm25.return_value = []
        mock_vector.return_value = []
        mock_rrf.return_value = []
        mock_rerank.side_effect = RuntimeError("rerank failed")
        with pytest.raises(RuntimeError, match="Error while reranking"):
            retrieval("query", MagicMock())

    @patch("src.retrieval.pipeline.rerank")
    @patch("src.retrieval.pipeline.vector_search")
    @patch("src.retrieval.pipeline.bm25_search")
    def test_error_stops_pipeline_early(self, mock_bm25, mock_vector, mock_rerank):
        """If BM25 fails, vector search and rerank should NOT be called."""
        mock_bm25.side_effect = RuntimeError("fail")
        with pytest.raises(RuntimeError):
            retrieval("query", MagicMock())
        mock_vector.assert_not_called()
        mock_rerank.assert_not_called()
