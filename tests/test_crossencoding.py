import unittest
from unittest.mock import patch, MagicMock
from src.retrieval.cross_encoder import rerank

class TestRerank(unittest.TestCase):
    @patch('src.retrieval.cross_encoder._get_model')
    def test_rerank(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        mock_get_model.return_value = mock_model

        query = "What is the capital of France?"
        chunks = [{"chunk_id": i, "text": f"Chunk {i}"} for i in range(10)]
        top_k = 5

        result = rerank(query, chunks, top_k)

        expected_pairs = [(query, chunk) for chunk in chunks]
        mock_get_model.assert_called_once()
        mock_model.predict.assert_called_once_with(expected_pairs)
        self.assertEqual(len(result), top_k)

    @patch('src.retrieval.cross_encoder._get_model')
    def test_rerank_empty_query(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        mock_get_model.return_value = mock_model

        query = ""
        chunks = [{"chunk_id": i, "text": f"Chunk {i}"} for i in range(10)]
        top_k = 5

        result = rerank(query, chunks, top_k)

        expected_pairs = [(query, chunk) for chunk in chunks]
        mock_get_model.assert_called_once()
        mock_model.predict.assert_called_once_with(expected_pairs)
        self.assertEqual(len(result), top_k)

    @patch('src.retrieval.cross_encoder._get_model')
    def test_rerank_invalid_top_k(self, mock_get_model):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        query = "What is the capital of France?"
        chunks = [{"chunk_id": i, "text": f"Chunk {i}"} for i in range(10)]
        top_k = -1

        with self.assertRaises(ValueError):
            rerank(query, chunks, top_k)

        mock_get_model.assert_not_called()
        mock_model.predict.assert_not_called()

if __name__ == '__main__':
    unittest.main()