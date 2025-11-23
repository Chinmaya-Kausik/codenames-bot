"""
Clue vocabulary for bridging word and vector representations.

This module provides the ClueVocab class that stores both vector embeddings
and optional word tokens, enabling conversion between the two representations.
"""

from __future__ import annotations

from typing import Optional
import torch


class ClueVocab:
    """
    Vocabulary of clues with both vector and word representations.

    Stores K clue vectors (and optionally tokens) that can be used by both
    word-based and vector-based environments. The vocabulary can be:
    - Frozen for the duration of a run (for stationarity)
    - Resampled between runs (for diversity)

    Attributes:
        vectors: [K, D] tensor of clue embeddings
        tokens: Optional list of K word strings
        dim: Embedding dimension D
        size: Vocabulary size K
        device: Device tensors are stored on
    """

    def __init__(
        self,
        vectors: torch.Tensor,
        tokens: Optional[list[str]] = None,
        device: Optional[torch.device | str] = None
    ):
        """
        Initialize clue vocabulary.

        Args:
            vectors: [K, D] tensor of clue embeddings
            tokens: Optional list of K word strings
            device: Device to store tensors on

        Raises:
            ValueError: If vectors and tokens have mismatched sizes
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D vectors tensor, got shape {vectors.shape}")

        if device is None:
            from utils.device import get_device
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        self.vectors = vectors.clone().to(self.device)
        self.size, self.dim = vectors.shape
        self._normalized_vectors = self._normalize(self.vectors)

        if tokens is not None:
            if len(tokens) != self.size:
                raise ValueError(
                    f"Token count {len(tokens)} doesn't match vector count {self.size}"
                )
            self.tokens = list(tokens)
            # Fast lookup to avoid O(K) scans when snapping via words
            self._token_to_index = {token: idx for idx, token in enumerate(self.tokens)}
        else:
            self.tokens = None
            self._token_to_index = None

    def snap(self, clue_vec: torch.Tensor) -> int | torch.Tensor:
        """
        Find nearest clue in vocabulary (reality layer snapping).

        Uses cosine similarity to find the closest vocabulary entry.

        Args:
            clue_vec: [D] or [B, D] clue vector(s) to snap

        Returns:
            Clue index (int) or [B] tensor of indices
        """
        clue_vec = clue_vec.to(self.device)

        if clue_vec.ndim == 1:
            return self._snap_single(clue_vec)

        # Batch snapping without Python loops
        clue_norm = self._normalize(clue_vec)
        similarities = clue_norm @ self._normalized_vectors.T
        return torch.argmax(similarities, dim=1).to(torch.int32)

    def _snap_single(self, clue_vec: torch.Tensor) -> int:
        """
        Snap a single clue vector to nearest vocabulary entry.

        Args:
            clue_vec: [D] clue vector

        Returns:
            Nearest clue index
        """
        clue_norm = self._normalize(clue_vec.unsqueeze(0))[0]
        similarities = self._normalized_vectors @ clue_norm
        return int(torch.argmax(similarities).item())

    def get_word(self, clue_index: int) -> str:
        """
        Get word token for a clue index.

        Args:
            clue_index: Index into vocabulary

        Returns:
            Word token

        Raises:
            ValueError: If tokens not available or index out of range
        """
        if self.tokens is None:
            raise ValueError("Tokens not available in this vocabulary")
        if not 0 <= clue_index < self.size:
            raise ValueError(f"Clue index {clue_index} out of range [0, {self.size})")

        return self.tokens[clue_index]

    def get_vector(self, clue_index: int) -> torch.Tensor:
        """
        Get vector embedding for a clue index.

        Args:
            clue_index: Index into vocabulary

        Returns:
            [D] vector embedding

        Raises:
            ValueError: If index out of range
        """
        if not 0 <= clue_index < self.size:
            raise ValueError(f"Clue index {clue_index} out of range [0, {self.size})")

        return self.vectors[clue_index].clone()

    def get_words_batch(self, clue_indices: torch.Tensor) -> list[str]:
        """
        Get word tokens for a batch of clue indices.

        Args:
            clue_indices: [B] tensor of indices

        Returns:
            List of B word tokens
        """
        if self.tokens is None:
            raise ValueError("Tokens not available in this vocabulary")

        clue_indices = clue_indices.to(self.device)
        return [self.tokens[idx.item()] for idx in clue_indices]

    def get_vectors_batch(self, clue_indices: torch.Tensor) -> torch.Tensor:
        """
        Get vector embeddings for a batch of clue indices.

        Args:
            clue_indices: [B] tensor of indices

        Returns:
            [B, D] tensor of vectors
        """
        clue_indices = clue_indices.to(self.device)
        return self.vectors[clue_indices].clone()

    @staticmethod
    def from_word_pool(
        words: list[str],
        embedding_model,
        max_size: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None
    ) -> ClueVocab:
        """
        Create vocabulary from a word pool using an embedding model.

        Args:
            words: List of candidate clue words
            embedding_model: Model with encode() method (e.g., SentenceTransformer)
            max_size: Maximum vocabulary size (if None, uses all words)
            seed: Random seed for sampling words
            device: Device to store tensors on

        Returns:
            ClueVocab instance
        """
        import random

        # Sample words if needed
        if max_size is not None and len(words) > max_size:
            rng = random.Random(seed)
            words = rng.sample(words, max_size)

        # Encode words to tensors
        if device is None:
            from utils.device import get_device
            device = get_device()

        vectors = embedding_model.encode(words, convert_to_tensor=True, device=str(device))

        return ClueVocab(vectors, tokens=words, device=device)

    @staticmethod
    def from_random_vectors(
        size: int,
        dim: int,
        seed: Optional[int] = None,
        normalize: bool = True,
        device: Optional[torch.device | str] = None
    ) -> ClueVocab:
        """
        Create vocabulary with random vectors (no tokens).

        Useful for testing or purely vector-based experiments.

        Args:
            size: Vocabulary size K
            dim: Embedding dimension D
            seed: Random seed
            normalize: Whether to normalize vectors to unit length
            device: Device to store tensors on

        Returns:
            ClueVocab instance with no tokens
        """
        if device is None:
            from utils.device import get_device
            device = get_device()
        device = torch.device(device) if isinstance(device, str) else device

        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        vectors = torch.randn(size, dim, device=device, generator=generator)

        if normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=1)

        return ClueVocab(vectors, tokens=None, device=device)

    @staticmethod
    def _normalize(vectors: torch.Tensor) -> torch.Tensor:
        """
        Normalize vectors with numerical stability.

        Args:
            vectors: Tensor of shape (..., D)

        Returns:
            Normalized tensor with same shape.
        """
        return torch.nn.functional.normalize(vectors, dim=-1)

    def to(self, device: torch.device | str) -> ClueVocab:
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
        self._normalized_vectors = self._normalized_vectors.to(device)
        return self
