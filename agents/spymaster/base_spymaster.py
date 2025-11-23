"""
Base class for spymaster agents in multi-agent Codenames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


@dataclass
class SpymasterParams:
    """
    Parameters for spymaster agents.

    Attributes:
        n_candidate_clues: Number of candidate clues to evaluate
        risk_tolerance: Higher values = more aggressive multi-word clues
        opponent_penalty: Penalty multiplier for opponent word similarity
        neutral_penalty: Penalty multiplier for neutral word similarity
        assassin_penalty: Penalty multiplier for assassin word similarity
        seed: Random seed for reproducibility
        embedding_model_name: Name of embedding model (e.g., "all-MiniLM-L6-v2")
        clue_word_pool: Custom list of words to use as candidate clues
    """
    n_candidate_clues: int = 50
    risk_tolerance: float = 2.0
    opponent_penalty: float = 2.0
    neutral_penalty: float = 1.0
    assassin_penalty: float = 10.0
    seed: Optional[int] = None
    embedding_model_name: Optional[str] = None
    clue_word_pool: Optional[list[str]] = None


class BaseSpymaster(ABC):
    """
    Abstract base class for spymaster agents.

    Spymasters observe the board with colors visible and give clues
    to their team's guesser.
    """

    def __init__(self, team: str, params: Optional[SpymasterParams] = None):
        """
        Initialize spymaster agent.

        Args:
            team: Team this spymaster plays for ("red" or "blue")
            params: SpymasterParams with configuration
        """
        self.team = team
        self.params = params if params is not None else SpymasterParams()

    @abstractmethod
    def get_clue(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Get clue action from observation.

        Args:
            obs: Agent-specific observation dict from environment containing:
                - "words": List of lists of board words (if WordBatchEnv)
                - "board_vectors": [B, N, D] tile embeddings (if VectorBatchEnv)
                - "colors": [B, N] color array (spymasters can see this)
                - "revealed": [B, N] revealed mask
                - "role_encoding": [B, 4] role encoding
                - "current_team": [B] current team
                - "phase": [B] current phase

        Returns:
            Action dict in one of these formats:
                - {"clue": List[str], "clue_number": np.ndarray([B])}  # Word clues
                - {"clue_vec": np.ndarray([B, D]), "clue_number": np.ndarray([B])}  # Vector clues
        """
        pass

    def reset(self) -> None:
        """Reset agent state for a new game (optional)."""
        pass
