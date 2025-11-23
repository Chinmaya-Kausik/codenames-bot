"""
Vector view for mapping tile IDs to embedding vectors.

This module provides the VectorView class that translates between tile indices
and vector embeddings for vector-based Codenames environments.
"""

from __future__ import annotations

from typing import Optional
import torch


class VectorView:
    """
    Maps tile IDs to embedding vectors for vector-based Codenames.

    Provides mapping from tile indices (0..N-1) to D-dimensional vectors.

    Attributes:
        vectors: [N, D] tensor of tile embeddings
        board_size: Number of tiles (N)
        embedding_dim: Embedding dimension (D)
        device: Device tensors are stored on
    """

    def __init__(self, vectors: torch.Tensor, device: Optional[torch.device | str] = None):
        """
        Initialize vector view.

        Args:
            vectors: [N, D] tensor of tile embeddings
            device: Device to store tensors on (cpu/cuda/mps)

        Raises:
            ValueError: If vectors is not 2D
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {vectors.shape}")

        if device is None:
            from utils.device import get_device
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        self.vectors = vectors.clone().to(self.device)
        self.board_size, self.embedding_dim = vectors.shape

    def get_vector(self, tile_index: int) -> torch.Tensor:
        """
        Get vector for a tile index.

        Args:
            tile_index: Tile index (0..N-1)

        Returns:
            [D] vector embedding

        Raises:
            IndexError: If index out of range
        """
        return self.vectors[tile_index].clone()

    def get_all_vectors(self) -> torch.Tensor:
        """
        Get all tile vectors.

        Returns:
            [N, D] tensor of all vectors
        """
        return self.vectors.clone()

    def get_unrevealed_vectors(self, revealed: torch.Tensor) -> torch.Tensor:
        """
        Get vectors for unrevealed tiles.

        Args:
            revealed: [N] bool tensor indicating revealed status

        Returns:
            [M, D] tensor of unrevealed vectors (M <= N)
        """
        revealed = revealed.to(self.device)
        return self.vectors[~revealed].clone()

    def get_unrevealed_indices(self, revealed: torch.Tensor) -> torch.Tensor:
        """
        Get indices of unrevealed tiles.

        Args:
            revealed: [N] bool tensor indicating revealed status

        Returns:
            [M] tensor of unrevealed tile indices
        """
        revealed = revealed.to(self.device)
        return torch.nonzero(~revealed, as_tuple=False).squeeze(-1)

    @staticmethod
    def create_random(
        board_size: int = 25,
        embedding_dim: int = 384,
        seed: Optional[int] = None,
        normalize: bool = True,
        device: Optional[torch.device | str] = None
    ) -> VectorView:
        """
        Create a random vector view with random embeddings.

        Useful for testing or synthetic experiments.

        Args:
            board_size: Number of tiles
            embedding_dim: Embedding dimension
            seed: Random seed
            normalize: Whether to normalize vectors to unit length
            device: Device to store tensors on

        Returns:
            VectorView instance
        """
        if device is None:
            from utils.device import get_device
            device = get_device()
        device = torch.device(device) if isinstance(device, str) else device

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        vectors = torch.randn(board_size, embedding_dim, device=device, generator=generator)

        if normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=1)

        return VectorView(vectors, device=device)

    @staticmethod
    def from_words(
        words: list[str],
        embedding_model,
        device: Optional[torch.device | str] = None
    ) -> VectorView:
        """
        Create vector view by embedding words.

        Args:
            words: List of word strings
            embedding_model: Model with encode() method (e.g., SentenceTransformer)
            device: Device to store tensors on

        Returns:
            VectorView instance
        """
        if device is None:
            from utils.device import get_device
            device = get_device()

        vectors = embedding_model.encode(words, convert_to_tensor=True, device=str(device))
        return VectorView(vectors, device=device)

    @staticmethod
    def create_batched(
        batch_size: int,
        board_size: int = 25,
        embedding_dim: int = 384,
        seed: Optional[int] = None,
        normalize: bool = True,
        device: Optional[torch.device | str] = None
    ) -> torch.Tensor:
        """
        Create batched vector views for vectorized environments.

        Args:
            batch_size: Number of views (B)
            board_size: Number of tiles per view (N)
            embedding_dim: Embedding dimension (D)
            seed: Random seed
            normalize: Whether to normalize vectors
            device: Device to store tensors on

        Returns:
            [B, N, D] tensor of batched vectors
        """
        if device is None:
            from utils.device import get_device
            device = get_device()
        device = torch.device(device) if isinstance(device, str) else device

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        vectors = torch.randn(batch_size, board_size, embedding_dim, device=device, generator=generator)

        if normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=2)

        return vectors

    def to(self, device: torch.device | str) -> VectorView:
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (cpu/cuda/mps)

        Returns:
            Self (for method chaining)
        """
        device = torch.device(device) if isinstance(device, str) else device
        if device == self.device:
            return self

        self.device = device
        self.vectors = self.vectors.to(device)
        return self
