"""Tests for generation and citation validation."""

import pytest

from src.generation.validator import extract_citations, validate_citations


class TestCitationExtraction:
    """Test citation parsing from answers."""
    
    def test_extract_single_citation(self):
        """Test extracting one citation."""
        answer = "The result is 42 [SOURCE 0]."
        citations = extract_citations(answer)
        assert citations == [0]
    
    def test_extract_multiple_citations(self):
        """Test extracting multiple citations."""
        answer = "Claim one [SOURCE 0] and claim two [SOURCE 1]."
        citations = extract_citations(answer)
        assert citations == [0, 1]
    
    def test_extract_duplicate_citations(self):
        """Test that duplicates are deduplicated."""
        answer = "[SOURCE 0] and again [SOURCE 0]."
        citations = extract_citations(answer)
        assert citations == [0]
    
    def test_extract_no_citations(self):
        """Test empty result when no citations."""
        answer = "No citations here."
        citations = extract_citations(answer)
        assert citations == []


class TestCitationValidation:
    """Test citation validation logic."""
    
    def test_valid_citations_pass(self):
        """Test that valid citations pass."""
        answer = "Claim [SOURCE 0] and [SOURCE 1]."
        is_valid, _, citations = validate_citations(answer, max_chunk_index=2)
        assert is_valid
        assert citations == [0, 1]
    
    def test_invalid_citation_raises(self):
        """Test that out-of-range citation raises error."""
        answer = "Claim [SOURCE 99]."
        with pytest.raises(ValueError):
            validate_citations(answer, max_chunk_index=2)
    
    def test_no_citations_raises(self):
        """Test that missing citations raises error."""
        answer = "No citations here."
        with pytest.raises(ValueError):
            validate_citations(answer, max_chunk_index=2)
