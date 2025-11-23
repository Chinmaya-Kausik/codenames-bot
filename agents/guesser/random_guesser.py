"""
Random baseline guesser agent.
"""

from __future__ import annotations

import random
from typing import Optional, Any
import torch

from agents.guesser.base_guesser import BaseGuesser, GuesserParams


class RandomGuesser(BaseGuesser):
    """
    Simple random baseline guesser.

    Makes random guesses from unrevealed words, with 20% chance to end turn early.
    """

    def __init__(self, team: str, params: Optional[GuesserParams] = None):
        """
        Initialize random guesser.

        Args:
            team: Team this guesser plays for
            params: GuesserParams (seed is used, other params ignored)
        """
        super().__init__(team, params)
        self.generator = torch.Generator()
        if self.params.seed is not None:
            self.generator.manual_seed(self.params.seed)

    def get_guess(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Make a random guess from unrevealed words using vectorized PyTorch operations.

        Args:
            obs: Agent observation from environment

        Returns:
            Action dict with random tile indices
        """
        # Get revealed mask and convert to torch if needed
        revealed = torch.as_tensor(obs["revealed"], dtype=torch.bool)  # [B, N]
        batch_size, board_size = revealed.shape
        device = revealed.device

        # Create random priority scores for each tile [B, N]
        # Generate on CPU with seeded generator, then move to device
        priorities = torch.rand(batch_size, board_size, generator=self.generator).to(device)

        # 20% chance to end turn early - favor first unrevealed tile
        early_exit = torch.rand(batch_size, generator=self.generator).to(device) < 0.2  # [B]

        # For early exit games, give first position highest priority
        position_bias = torch.arange(board_size, 0, -1, dtype=torch.float32, device=device)  # [N] descending
        position_bias = position_bias.unsqueeze(0).expand(batch_size, -1)  # [B, N]

        # Apply position bias for early exit games
        priorities = torch.where(
            early_exit.unsqueeze(1),
            position_bias,
            priorities
        )

        # Mask out revealed tiles with very low priority
        priorities = torch.where(revealed, torch.tensor(-1e9, device=device), priorities)

        # Pick highest priority unrevealed tile per game
        tile_indices = torch.argmax(priorities, dim=1).to(torch.int32)  # [B]

        # Handle edge case: all tiles revealed (shouldn't happen in practice)
        all_revealed = revealed.all(dim=1)  # [B]
        tile_indices = torch.where(all_revealed, torch.zeros_like(tile_indices), tile_indices)

        return {"tile_index": tile_indices}
