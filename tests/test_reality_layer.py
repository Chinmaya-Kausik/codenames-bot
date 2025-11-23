"""
Tests for core.reality_layer module.
"""

import torch
import pytest

from core.clue_vocab import ClueVocab
from core.reality_layer import RealityLayer, create_reality_layer_from_random


# Use CPU for tests to avoid device mismatch issues
DEVICE = torch.device("cpu")


def test_reality_layer_enabled():
    """Test reality layer with snapping enabled."""
    # Create simple vocab
    vectors = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    tokens = ["HORIZONTAL", "VERTICAL"]

    vocab = ClueVocab(vectors, tokens, device=DEVICE)
    layer = RealityLayer(vocab, enabled=True)

    assert layer.enabled

    # Test snapping
    clue_vecs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    clue_numbers = torch.tensor([2, 3], dtype=torch.int32)

    indices, final_vecs, final_numbers = layer.apply(clue_vecs, clue_numbers)

    assert torch.equal(indices, torch.tensor([0, 1], dtype=torch.int32, device=DEVICE))
    assert torch.allclose(final_vecs[0], torch.tensor([1.0, 0.0], device=DEVICE))
    assert torch.allclose(final_vecs[1], torch.tensor([0.0, 1.0], device=DEVICE))
    assert torch.equal(final_numbers, clue_numbers.to(DEVICE))


def test_reality_layer_disabled():
    """Test reality layer with snapping disabled."""
    vectors = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    vocab = ClueVocab(vectors, device=DEVICE)
    layer = RealityLayer(vocab, enabled=False)

    assert not layer.enabled

    # Test without snapping
    clue_vecs = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    clue_numbers = torch.tensor([2, 3], dtype=torch.int32)

    indices, final_vecs, final_numbers = layer.apply(clue_vecs, clue_numbers)

    # Indices should be -1 (disabled)
    assert torch.all(indices == -1)

    # Vectors should be unchanged
    assert torch.allclose(final_vecs.to(DEVICE), clue_vecs.to(DEVICE))
    assert torch.equal(final_numbers, clue_numbers.to(DEVICE))


def test_get_words_enabled():
    """Test getting words from indices when enabled."""
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tokens = ["WORD_A", "WORD_B"]

    vocab = ClueVocab(vectors, tokens, device=DEVICE)
    layer = RealityLayer(vocab, enabled=True)

    indices = torch.tensor([0, 1, 0], dtype=torch.int32)
    words = layer.get_words(indices)

    assert words == ["WORD_A", "WORD_B", "WORD_A"]


def test_get_words_disabled():
    """Test getting words when disabled returns None."""
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tokens = ["WORD_A", "WORD_B"]

    vocab = ClueVocab(vectors, tokens, device=DEVICE)
    layer = RealityLayer(vocab, enabled=False)

    indices = torch.tensor([0, 1], dtype=torch.int32)
    words = layer.get_words(indices)

    assert words is None


def test_get_words_no_tokens():
    """Test getting words when vocab has no tokens returns None."""
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    vocab = ClueVocab(vectors, tokens=None, device=DEVICE)
    layer = RealityLayer(vocab, enabled=True)

    indices = torch.tensor([0, 1], dtype=torch.int32)
    words = layer.get_words(indices)

    assert words is None


def test_set_enabled():
    """Test toggling enabled/disabled."""
    vectors = torch.randn(10, 5)
    vocab = ClueVocab(vectors, device=DEVICE)

    layer = RealityLayer(vocab, enabled=True)
    assert layer.enabled

    layer.set_enabled(False)
    assert not layer.enabled

    layer.set_enabled(True)
    assert layer.enabled


def test_create_reality_layer_from_random():
    """Test creating reality layer with random vocab."""
    vocab_size = 100
    embedding_dim = 384

    layer = create_reality_layer_from_random(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        enabled=True,
        seed=42,
        device=DEVICE
    )

    assert layer.enabled
    assert layer.clue_vocab.size == vocab_size
    assert layer.clue_vocab.dim == embedding_dim
    assert layer.clue_vocab.tokens is None


def test_batch_consistency():
    """Test that batched operations are consistent."""
    vocab_size = 50
    embedding_dim = 10
    batch_size = 5

    layer = create_reality_layer_from_random(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        enabled=True,
        seed=42,
        device=DEVICE
    )

    # Create random clue vectors
    clue_vecs = torch.randn(batch_size, embedding_dim)
    clue_numbers = torch.randint(1, 4, size=(batch_size,), dtype=torch.int32)

    indices, final_vecs, final_numbers = layer.apply(clue_vecs, clue_numbers)

    # Check shapes
    assert indices.shape == (batch_size,)
    assert final_vecs.shape == (batch_size, embedding_dim)
    assert final_numbers.shape == (batch_size,)

    # Check that snapped vectors are actually from vocab
    for i in range(batch_size):
        expected_vec = layer.clue_vocab.get_vector(indices[i].item())
        assert torch.allclose(final_vecs[i], expected_vec)


def test_preserves_clue_numbers():
    """Test that reality layer doesn't modify clue numbers."""
    vectors = torch.randn(10, 5)
    vocab = ClueVocab(vectors, device=DEVICE)
    layer = RealityLayer(vocab, enabled=True)

    clue_vecs = torch.randn(3, 5)
    clue_numbers = torch.tensor([1, 2, 3], dtype=torch.int32)

    _, _, final_numbers = layer.apply(clue_vecs, clue_numbers)

    assert torch.equal(final_numbers, clue_numbers.to(DEVICE))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
