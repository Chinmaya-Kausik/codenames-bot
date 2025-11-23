"""
Reality layer for mapping continuous agent outputs to discrete clue vocabulary.

This module provides utilities for "snapping" continuous vectors to a fixed
ClueVocab, preserving the combinatorial hardness of Codenames while allowing
continuous policy outputs.
"""

from __future__ import annotations

from typing import Optional
import torch

from core.clue_vocab import ClueVocab


class RealityLayer:
    """
    Reality layer for snapping continuous outputs to discrete vocabulary.

    The reality layer can be toggled on/off:
    - ON: Snap continuous vectors to nearest vocab entry (combinatorial mode)
    - OFF: Use continuous vectors directly (for experiments/debugging)

    Attributes:
        clue_vocab: ClueVocab to snap to
        enabled: Whether snapping is enabled
    """

    def __init__(self, clue_vocab: ClueVocab, enabled: bool = True):
        """
        Initialize reality layer.

        Args:
            clue_vocab: Vocabulary to snap to
            enabled: Whether to enable snapping (default True)
        """
        self.clue_vocab = clue_vocab
        self.enabled = enabled

    def apply(
        self,
        clue_vecs: torch.Tensor,
        clue_numbers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply reality layer to clue actions.

        If enabled, snaps continuous vectors to nearest vocab entry.
        If disabled, returns vectors unchanged with dummy indices.

        Args:
            clue_vecs: [B, D] continuous clue vectors
            clue_numbers: [B] clue numbers

        Returns:
            Tuple of:
                - clue_indices: [B] indices into vocab (or -1 if disabled)
                - final_vecs: [B, D] final vectors (snapped if enabled)
                - clue_numbers: [B] clue numbers (unchanged)
        """
        clue_vecs = clue_vecs.to(self.clue_vocab.device)
        clue_numbers = clue_numbers.to(self.clue_vocab.device)

        if not self.enabled:
            # Reality layer off: use continuous vectors directly
            batch_size = clue_vecs.shape[0]
            clue_indices = torch.full((batch_size,), -1, dtype=torch.int32, device=self.clue_vocab.device)
            return clue_indices, clue_vecs.clone(), clue_numbers

        # Reality layer on: snap to vocab
        clue_indices = self.clue_vocab.snap(clue_vecs)
        final_vecs = self.clue_vocab.get_vectors_batch(clue_indices)

        return clue_indices, final_vecs, clue_numbers

    def get_words(self, clue_indices: torch.Tensor) -> Optional[list[str]]:
        """
        Get word tokens for clue indices.

        Args:
            clue_indices: [B] indices into vocab

        Returns:
            List of B words, or None if vocab has no tokens or layer disabled
        """
        if not self.enabled or self.clue_vocab.tokens is None:
            return None

        try:
            return self.clue_vocab.get_words_batch(clue_indices)
        except (ValueError, IndexError):
            return None

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable reality layer snapping.

        Args:
            enabled: Whether to enable snapping
        """
        self.enabled = enabled

    def to(self, device: torch.device | str) -> RealityLayer:
        """
        Move reality layer to specified device.

        Args:
            device: Target device

        Returns:
            Self (for method chaining)
        """
        self.clue_vocab.to(device)
        return self


def create_reality_layer_from_words(
    words: list[str],
    embedding_model,
    max_vocab_size: Optional[int] = 1000,
    enabled: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None
) -> RealityLayer:
    """
    Create a reality layer from a word pool.

    Convenience function for creating a ClueVocab and RealityLayer together.

    Args:
        words: List of candidate clue words
        embedding_model: Model with encode() method (e.g., SentenceTransformer)
        max_vocab_size: Maximum vocabulary size
        enabled: Whether reality layer is enabled
        seed: Random seed for word sampling
        device: Device to store tensors on

    Returns:
        RealityLayer instance
    """
    clue_vocab = ClueVocab.from_word_pool(
        words, embedding_model, max_size=max_vocab_size, seed=seed, device=device
    )
    return RealityLayer(clue_vocab, enabled=enabled)


def create_reality_layer_from_random(
    vocab_size: int = 1000,
    embedding_dim: int = 384,
    enabled: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None
) -> RealityLayer:
    """
    Create a reality layer with random vectors (no words).

    Useful for testing or purely vector-based experiments.

    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of embeddings
        enabled: Whether reality layer is enabled
        seed: Random seed
        device: Device to store tensors on

    Returns:
        RealityLayer instance
    """
    clue_vocab = ClueVocab.from_random_vectors(
        size=vocab_size, dim=embedding_dim, seed=seed, device=device
    )
    return RealityLayer(clue_vocab, enabled=enabled)
