"""
Random baseline spymaster agent.
"""

from __future__ import annotations

import random
from typing import Optional, Any
import torch

from agents.spymaster.base_spymaster import BaseSpymaster, SpymasterParams


class RandomSpymaster(BaseSpymaster):
    """
    Simple random baseline spymaster.

    Gives random clues with random numbers between 1-3.
    """

    def __init__(self, team: str, params: Optional[SpymasterParams] = None):
        """
        Initialize random spymaster.

        Args:
            team: Team this spymaster plays for
            params: SpymasterParams (seed is used, other params ignored)
        """
        super().__init__(team, params)
        self.rng = random.Random(self.params.seed)

    def get_clue(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Give a random clue.

        Args:
            obs: Agent observation from environment

        Returns:
            Action dict with random clue word and number 1-3
        """
        # Determine batch size from observation
        if "words" in obs and obs["words"] is not None:
            # WordBatchEnv
            batch_size = len(obs["words"])
        elif "board_vectors" in obs:
            # VectorBatchEnv
            batch_size = obs["board_vectors"].shape[0]
        else:
            batch_size = 1

        # Generate random clue words
        clue_words = [
            self.rng.choice(["RANDOM", "TEST", "EXAMPLE", "WORD", "CLUE"])
            for _ in range(batch_size)
        ]

        # Random numbers between 1 and 3
        clue_numbers = torch.tensor([self.rng.randint(1, 3) for _ in range(batch_size)], dtype=torch.int32)

        return {
            "clue": clue_words,
            "clue_number": clue_numbers
        }
