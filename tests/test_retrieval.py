"""Tests for retrieval pipeline."""

import pytest

from src.retrieval.hybrid_fusion import reciprocal_rank_fusion


class TestHybridFusion:
    """Test RRF fusion logic."""
    
    def test_fusion_combines_results(self):
        """Test that fusion includes chunks from both sources."""
        vector = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        bm25 = [("b", 0.85), ("d", 0.75), ("e", 0.65)]
        
        fused = reciprocal_rank_fusion(vector, bm25, k=60)
        
        # Should have 5 unique chunks
        assert len(fused) == 5
        
        # Chunk 'b' appears in both, should have higher score
        b_score = next(score for cid, score in fused if cid == "b")
        e_score = next(score for cid, score in fused if cid == "e")
        assert b_score > e_score  # 'b' ranked better in both lists
    
    def test_fusion_empty_inputs(self):
        """Test fusion with empty inputs."""
        fused = reciprocal_rank_fusion([], [("a", 1.0)])
        assert len(fused) == 1
        
        fused = reciprocal_rank_fusion([], [])
        assert len(fused) == 0


class TestReranker:
    """Test cross-encoder reranking."""
    
    def test_rerank_orders_by_score(self):
        """Test that reranker sorts by score."""
        # Note: This requires the model loaded, may be slow
        # Consider mocking for unit tests
        pass
