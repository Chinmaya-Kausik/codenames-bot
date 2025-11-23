"""
Tests for views.vector_view module.
"""

import torch
import pytest

from views.vector_view import VectorView


# Use CPU for tests to avoid device mismatch issues
DEVICE = torch.device("cpu")


def test_vector_view_initialization():
    """Test basic initialization."""
    vectors = torch.randn(25, 384)
    view = VectorView(vectors, device=DEVICE)

    assert view.board_size == 25
    assert view.embedding_dim == 384
    assert view.vectors.shape == (25, 384)


def test_invalid_shape():
    """Test error with invalid vector shape."""
    # 1D array
    with pytest.raises(ValueError):
        VectorView(torch.randn(10), device=DEVICE)

    # 3D array
    with pytest.raises(ValueError):
        VectorView(torch.randn(5, 5, 10), device=DEVICE)


def test_get_vector():
    """Test getting vector by index."""
    vectors = torch.randn(10, 5, device=DEVICE)
    view = VectorView(vectors, device=DEVICE)

    vec = view.get_vector(3)

    assert vec.shape == (5,)
    assert torch.allclose(vec, vectors[3])

    # Check it's a clone
    vec[0] = 999
    assert view.vectors[3, 0] != 999


def test_get_all_vectors():
    """Test getting all vectors."""
    vectors = torch.randn(10, 5, device=DEVICE)
    view = VectorView(vectors, device=DEVICE)

    all_vecs = view.get_all_vectors()

    assert all_vecs.shape == (10, 5)
    assert torch.allclose(all_vecs, vectors)

    # Check it's a clone
    all_vecs[0, 0] = 999
    assert view.vectors[0, 0] != 999


def test_get_unrevealed_vectors():
    """Test getting unrevealed vectors."""
    vectors = torch.randn(5, 3, device=DEVICE)
    view = VectorView(vectors, device=DEVICE)

    revealed = torch.tensor([False, True, False, True, False], device=DEVICE)
    unrevealed = view.get_unrevealed_vectors(revealed)

    assert unrevealed.shape == (3, 3)  # 3 unrevealed
    assert torch.allclose(unrevealed[0], vectors[0])
    assert torch.allclose(unrevealed[1], vectors[2])
    assert torch.allclose(unrevealed[2], vectors[4])


def test_get_unrevealed_indices():
    """Test getting unrevealed indices."""
    vectors = torch.randn(5, 3, device=DEVICE)
    view = VectorView(vectors, device=DEVICE)

    revealed = torch.tensor([False, True, False, True, False], device=DEVICE)
    indices = view.get_unrevealed_indices(revealed)

    assert torch.equal(indices, torch.tensor([0, 2, 4], device=DEVICE))


def test_create_random():
    """Test creating random vector view."""
    view = VectorView.create_random(
        board_size=25,
        embedding_dim=384,
        seed=42,
        normalize=True,
        device=DEVICE
    )

    assert view.board_size == 25
    assert view.embedding_dim == 384

    # Check normalization
    norms = torch.norm(view.vectors, dim=1)
    assert torch.allclose(norms, torch.ones(25, device=DEVICE), atol=1e-6)

    # Same seed should give same result
    view2 = VectorView.create_random(
        board_size=25,
        embedding_dim=384,
        seed=42,
        normalize=True,
        device=DEVICE
    )
    assert torch.allclose(view.vectors, view2.vectors)


def test_create_random_no_normalize():
    """Test creating random vectors without normalization."""
    view = VectorView.create_random(
        board_size=25,
        embedding_dim=384,
        seed=42,
        normalize=False,
        device=DEVICE
    )

    # Norms should NOT all be 1.0
    norms = torch.norm(view.vectors, dim=1)
    assert not torch.allclose(norms, torch.ones(25, device=DEVICE), atol=0.1)


def test_create_batched():
    """Test creating batched vectors."""
    batch_vecs = VectorView.create_batched(
        batch_size=10,
        board_size=25,
        embedding_dim=384,
        seed=42,
        normalize=True,
        device=DEVICE
    )

    assert batch_vecs.shape == (10, 25, 384)

    # Check normalization across all batches
    norms = torch.norm(batch_vecs, dim=2)
    assert torch.allclose(norms, torch.ones(10, 25, device=DEVICE), atol=1e-6)


def test_vectors_are_copied():
    """Test that vectors are cloned, not referenced."""
    original = torch.randn(5, 3, device=DEVICE)
    view = VectorView(original, device=DEVICE)

    # Modify original
    original[0, 0] = 999

    # View should be unchanged
    assert view.vectors[0, 0] != 999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
