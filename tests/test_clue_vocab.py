"""
Tests for core.clue_vocab module.
"""

import torch
import pytest

from core.clue_vocab import ClueVocab


# Use CPU for tests to avoid device mismatch issues
DEVICE = torch.device("cpu")


def test_clue_vocab_initialization():
    """Test basic initialization."""
    size = 100
    dim = 384

    vectors = torch.randn(size, dim)
    tokens = [f"WORD_{i}" for i in range(size)]

    vocab = ClueVocab(vectors, tokens, device=DEVICE)

    assert vocab.size == size
    assert vocab.dim == dim
    assert len(vocab.tokens) == size
    assert vocab.vectors.shape == (size, dim)


def test_clue_vocab_without_tokens():
    """Test initialization without tokens."""
    size = 50
    dim = 128

    vectors = torch.randn(size, dim)
    vocab = ClueVocab(vectors, tokens=None, device=DEVICE)

    assert vocab.size == size
    assert vocab.dim == dim
    assert vocab.tokens is None


def test_snap_single_vector():
    """Test snapping a single vector to nearest entry."""
    # Create simple vocab with known vectors
    vectors = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    vocab = ClueVocab(vectors, device=DEVICE)

    # Test snapping to each axis
    assert vocab.snap(torch.tensor([0.9, 0.1, 0.0])) == 0
    assert vocab.snap(torch.tensor([0.1, 0.9, 0.0])) == 1
    assert vocab.snap(torch.tensor([0.0, 0.1, 0.9])) == 2


def test_snap_batch():
    """Test snapping a batch of vectors."""
    vectors = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    vocab = ClueVocab(vectors, device=DEVICE)

    # Batch of 3 vectors
    batch_vecs = torch.tensor([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9]
    ])

    indices = vocab.snap(batch_vecs)

    assert isinstance(indices, torch.Tensor)
    assert indices.shape == (3,)
    assert torch.equal(indices, torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE))


def test_get_word():
    """Test getting word by index."""
    vectors = torch.randn(10, 5)
    tokens = [f"WORD_{i}" for i in range(10)]

    vocab = ClueVocab(vectors, tokens, device=DEVICE)

    assert vocab.get_word(0) == "WORD_0"
    assert vocab.get_word(5) == "WORD_5"
    assert vocab.get_word(9) == "WORD_9"

    # Test out of range
    with pytest.raises(ValueError):
        vocab.get_word(10)

    # Test vocab without tokens
    vocab_no_tokens = ClueVocab(vectors, tokens=None, device=DEVICE)
    with pytest.raises(ValueError):
        vocab_no_tokens.get_word(0)


def test_get_vector():
    """Test getting vector by index."""
    vectors = torch.randn(10, 5, device=DEVICE)
    vocab = ClueVocab(vectors, device=DEVICE)

    vec = vocab.get_vector(3)
    assert vec.shape == (5,)
    assert torch.allclose(vec, vectors[3])

    # Test out of range
    with pytest.raises(ValueError):
        vocab.get_vector(10)


def test_get_words_batch():
    """Test getting batch of words."""
    vectors = torch.randn(10, 5)
    tokens = [f"WORD_{i}" for i in range(10)]
    vocab = ClueVocab(vectors, tokens, device=DEVICE)

    indices = torch.tensor([0, 5, 9], dtype=torch.int32)
    words = vocab.get_words_batch(indices)

    assert words == ["WORD_0", "WORD_5", "WORD_9"]


def test_get_vectors_batch():
    """Test getting batch of vectors."""
    vectors = torch.randn(10, 5, device=DEVICE)
    vocab = ClueVocab(vectors, device=DEVICE)

    indices = torch.tensor([0, 5, 9], dtype=torch.int32)
    batch_vecs = vocab.get_vectors_batch(indices)

    assert batch_vecs.shape == (3, 5)
    assert torch.allclose(batch_vecs[0], vectors[0])
    assert torch.allclose(batch_vecs[1], vectors[5])
    assert torch.allclose(batch_vecs[2], vectors[9])


def test_from_random_vectors():
    """Test creating vocab from random vectors."""
    size = 100
    dim = 384

    vocab = ClueVocab.from_random_vectors(size, dim, seed=42, normalize=True, device=DEVICE)

    assert vocab.size == size
    assert vocab.dim == dim
    assert vocab.tokens is None

    # Check normalization
    norms = torch.norm(vocab.vectors, dim=1)
    assert torch.allclose(norms, torch.ones(size, device=DEVICE), atol=1e-6)


def test_mismatched_tokens_and_vectors():
    """Test error when tokens and vectors have different sizes."""
    vectors = torch.randn(10, 5)
    tokens = [f"WORD_{i}" for i in range(5)]  # Wrong size

    with pytest.raises(ValueError):
        ClueVocab(vectors, tokens, device=DEVICE)


def test_invalid_vector_shape():
    """Test error when vectors is not 2D."""
    vectors_1d = torch.randn(10)

    with pytest.raises(ValueError):
        ClueVocab(vectors_1d, device=DEVICE)

    vectors_3d = torch.randn(10, 5, 3)

    with pytest.raises(ValueError):
        ClueVocab(vectors_3d, device=DEVICE)


def test_cosine_similarity_snapping():
    """Test that snapping uses cosine similarity correctly."""
    # Create orthogonal vectors
    vectors = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    vocab = ClueVocab(vectors, device=DEVICE)

    # Vector close to first (but longer)
    test_vec = torch.tensor([10.0, 0.1])
    assert vocab.snap(test_vec) == 0

    # Vector close to second (but longer)
    test_vec = torch.tensor([0.1, 10.0])
    assert vocab.snap(test_vec) == 1

    # Vector at 45 degrees (should snap to one of them based on small differences)
    test_vec = torch.tensor([1.0, 1.0])
    result = vocab.snap(test_vec)
    assert result in [0, 1]  # Result is int, not tensor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
