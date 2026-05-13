from src.config import Config

def test_chunk_size():
    assert Config.CHUNK_SIZE == 350

def test_chunk_overlap():
    assert Config.CHUNK_OVERLAP == 75

def test_rrf_k():
    assert Config.RRF_K == 60

def test_top_k_rerank():
    assert Config.TOP_K_RERANK == 5
