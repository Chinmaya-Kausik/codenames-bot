"""
Word view for mapping tile IDs to word strings.

This module provides the WordView class that translates between tile indices
and word strings for word-based Codenames environments.
"""

from __future__ import annotations

from typing import Optional
import random


class WordView:
    """
    Maps tile IDs to word strings for word-based Codenames.

    Provides bidirectional mapping between tile indices (0..N-1) and words.

    Attributes:
        words: List of N word strings
        board_size: Number of tiles (N)
    """

    def __init__(self, words: list[str]):
        """
        Initialize word view.

        Args:
            words: List of word strings (one per tile)
        """
        self.words = [w.upper() for w in words]
        self.board_size = len(words)

        # Create reverse mapping
        self._word_to_index = {w: i for i, w in enumerate(self.words)}

    def get_word(self, tile_index: int) -> str:
        """
        Get word for a tile index.

        Args:
            tile_index: Tile index (0..N-1)

        Returns:
            Word string

        Raises:
            IndexError: If index out of range
        """
        return self.words[tile_index]

    def get_index(self, word: str) -> int:
        """
        Get tile index for a word.

        Args:
            word: Word string (case-insensitive)

        Returns:
            Tile index

        Raises:
            ValueError: If word not found
        """
        word_upper = word.upper()
        if word_upper not in self._word_to_index:
            raise ValueError(f"Word '{word}' not found in board")
        return self._word_to_index[word_upper]

    def get_all_words(self) -> list[str]:
        """
        Get all words on the board.

        Returns:
            List of N word strings
        """
        return self.words.copy()

    def get_unrevealed_words(self, revealed: list[bool]) -> list[str]:
        """
        Get list of unrevealed words.

        Args:
            revealed: List of N booleans indicating revealed status

        Returns:
            List of unrevealed words
        """
        return [w for w, r in zip(self.words, revealed) if not r]

    @staticmethod
    def create_random(
        word_pool: list[str],
        board_size: int = 25,
        seed: Optional[int] = None
    ) -> WordView:
        """
        Create a random word view by sampling from a word pool.

        Args:
            word_pool: Pool of candidate words
            board_size: Number of words to sample
            seed: Random seed

        Returns:
            WordView instance

        Raises:
            ValueError: If word pool too small
        """
        if len(word_pool) < board_size:
            raise ValueError(
                f"Word pool size {len(word_pool)} < board size {board_size}"
            )

        rng = random.Random(seed)
        words = rng.sample(word_pool, board_size)

        return WordView(words)

    @staticmethod
    def create_batched(
        word_pool: list[str],
        batch_size: int,
        board_size: int = 25,
        seed: Optional[int] = None
    ) -> list[WordView]:
        """
        Create multiple random word views for batched environments.

        Args:
            word_pool: Pool of candidate words
            batch_size: Number of views to create
            board_size: Number of words per view
            seed: Random seed

        Returns:
            List of B WordView instances
        """
        rng = random.Random(seed)
        views = []

        for _ in range(batch_size):
            # Use different seed for each view
            view_seed = rng.randint(0, 2**31 - 1)
            views.append(WordView.create_random(word_pool, board_size, view_seed))

        return views
