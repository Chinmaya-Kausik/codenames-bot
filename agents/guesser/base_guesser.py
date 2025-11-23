"""
Base class for guesser agents in multi-agent Codenames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


@dataclass
class GuesserParams:
    """
    Parameters for guesser agents.

    Attributes:
        similarity_threshold: Minimum similarity to guess a word
        confidence_threshold: Minimum confidence for last guess
        seed: Random seed for reproducibility
        embedding_model_name: Name of embedding model (e.g., "all-MiniLM-L6-v2")
    """
    similarity_threshold: float = 0.3
    confidence_threshold: float = 0.5
    seed: Optional[int] = None
    embedding_model_name: Optional[str] = None


class BaseGuesser(ABC):
    """
    Abstract base class for guesser agents.

    Guessers observe the board without colors and make guesses based
    on their spymaster's clue.
    """

    def __init__(self, team: str, params: Optional[GuesserParams] = None):
        """
        Initialize guesser agent.

        Args:
            team: Team this guesser plays for ("red" or "blue")
            params: GuesserParams with configuration
        """
        self.team = team
        self.params = params if params is not None else GuesserParams()

    @abstractmethod
    def get_guess(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Get guess action from observation.

        Args:
            obs: Agent-specific observation dict from environment containing:
                - "words": List of lists of board words (if WordBatchEnv)
                - "board_vectors": [B, N, D] tile embeddings (if VectorBatchEnv)
                - "colors": None (guessers can't see colors)
                - "revealed": [B, N] revealed mask
                - "current_clue": List[str] current clue words (if WordBatchEnv)
                - "current_clue_vec": [B, D] current clue vector (if VectorBatchEnv)
                - "current_clue_number": [B] clue number
                - "remaining_guesses": [B] guesses remaining
                - "role_encoding": [B, 4] role encoding
                - "current_team": [B] current team
                - "phase": [B] current phase

        Returns:
            Action dict in one of these formats:
                - {"word": List[str]}  # Word guesses
                - {"tile_index": np.ndarray([B])}  # Index guesses
        """
        pass

    def reset(self) -> None:
        """Reset agent state for a new game (optional)."""
        pass
