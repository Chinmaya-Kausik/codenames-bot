"""
Representation-agnostic batched game state for Codenames.

This module provides the core game logic that works with tile IDs, colors,
and revealed states, without any word or vector semantics.
"""

from __future__ import annotations

from typing import Optional
import torch


class GameState:
    """
    Batched, representation-agnostic Codenames game state.

    Manages B parallel games using PyTorch tensors. State is represented
    using tile IDs (0..N-1), colors, and revealed flags, with no dependence
    on word or vector representations.

    Attributes:
        batch_size: Number of parallel games (B)
        board_size: Number of tiles per board (N)
        device: Device tensors are stored on (cpu/cuda/mps)
        colors: [B, N] int tensor (0=red, 1=blue, 2=neutral, 3=assassin)
        revealed: [B, N] bool tensor
        current_team: [B] int tensor (0=red, 1=blue)
        phase: [B] int tensor (0=spymaster, 1=guesser)
        game_over: [B] bool tensor
        winner: [B] int tensor (-1=none, 0=red, 1=blue)
        remaining_guesses: [B] int tensor
        current_clue_index: [B] int tensor (index into ClueVocab, -1=none)
        current_clue_number: [B] int tensor
        turn_count: [B] int tensor
    """

    # Color constants
    RED = 0
    BLUE = 1
    NEUTRAL = 2
    ASSASSIN = 3

    # Phase constants
    SPYMASTER_PHASE = 0
    GUESSER_PHASE = 1

    def __init__(
        self,
        batch_size: int,
        board_size: int = 25,
        colors: Optional[torch.Tensor] = None,
        starting_teams: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None
    ):
        """
        Initialize batched game state.

        Args:
            batch_size: Number of parallel games
            board_size: Number of tiles per board (default 25 for 5x5)
            colors: [B, N] color assignments (if None, randomly generated)
            starting_teams: [B] starting teams (if None, randomly chosen)
            seed: Random seed for initialization
            device: Device to store tensors on (cpu/cuda/mps)
        """
        self.batch_size = batch_size
        self.board_size = board_size

        if device is None:
            from utils.device import get_device
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Initialize colors if not provided
        if colors is None:
            # Need to initialize current_team first for color generation
            if starting_teams is None:
                self.current_team = torch.randint(
                    0, 2, (batch_size,), dtype=torch.int32, device=self.device, generator=generator
                )
            else:
                self.current_team = starting_teams.clone().to(self.device)
            self.colors = self._generate_random_colors(generator)
        else:
            assert colors.shape == (batch_size, board_size)
            self.colors = colors.clone().to(self.device)
            if starting_teams is None:
                self.current_team = torch.randint(
                    0, 2, (batch_size,), dtype=torch.int32, device=self.device, generator=generator
                )
            else:
                self.current_team = starting_teams.clone().to(self.device)

        # Initialize all state tensors
        self.revealed = torch.zeros((batch_size, board_size), dtype=torch.bool, device=self.device)
        self.phase = torch.full((batch_size,), self.SPYMASTER_PHASE, dtype=torch.int32, device=self.device)
        self.game_over = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        self.winner = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
        self.remaining_guesses = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.current_clue_index = torch.full((batch_size,), -1, dtype=torch.int32, device=self.device)
        self.current_clue_number = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.turn_count = torch.zeros(batch_size, dtype=torch.int32, device=self.device)

    def _generate_random_colors(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Generate random color assignments for all boards using vectorized operations.

        Standard distribution: 36% first team, 32% second team, 28% neutral, 4% assassin.

        Args:
            generator: Random number generator

        Returns:
            [B, N] int tensor of color assignments
        """
        # Calculate counts
        first_count = max(1, int(self.board_size * 0.36))
        second_count = max(1, int(self.board_size * 0.32))
        assassin_count = max(1, int(self.board_size * 0.04))
        neutral_count = self.board_size - first_count - second_count - assassin_count

        # Precompute base layouts for red-first and blue-first games
        red_first_layout = torch.cat([
            torch.full((first_count,), self.RED, dtype=torch.int32, device=self.device),
            torch.full((second_count,), self.BLUE, dtype=torch.int32, device=self.device),
            torch.full((neutral_count,), self.NEUTRAL, dtype=torch.int32, device=self.device),
            torch.full((assassin_count,), self.ASSASSIN, dtype=torch.int32, device=self.device)
        ])

        blue_first_layout = torch.cat([
            torch.full((first_count,), self.BLUE, dtype=torch.int32, device=self.device),
            torch.full((second_count,), self.RED, dtype=torch.int32, device=self.device),
            torch.full((neutral_count,), self.NEUTRAL, dtype=torch.int32, device=self.device),
            torch.full((assassin_count,), self.ASSASSIN, dtype=torch.int32, device=self.device)
        ])

        # Stack layouts and select based on current_team [B, N]
        layouts = torch.stack([red_first_layout, blue_first_layout], dim=0)  # [2, N]
        colors = layouts[self.current_team]  # [B, N]

        # Batched shuffle using torch.rand + argsort
        # Generate random values for each position
        rand_vals = torch.rand((self.batch_size, self.board_size), device=self.device, generator=generator)
        perm_indices = torch.argsort(rand_vals, dim=1)  # [B, N]

        # Apply permutation using gather
        colors = torch.gather(colors, 1, perm_indices)

        return colors

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset all games to initial state.

        Args:
            seed: Random seed for reset
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Regenerate colors and starting teams
        self.current_team = torch.randint(
            0, 2, (self.batch_size,), dtype=torch.int32, device=self.device, generator=generator
        )
        self.colors = self._generate_random_colors(generator)

        # Reset all state
        self.revealed.fill_(False)
        self.phase.fill_(self.SPYMASTER_PHASE)
        self.game_over.fill_(False)
        self.winner.fill_(-1)
        self.remaining_guesses.fill_(0)
        self.current_clue_index.fill_(-1)
        self.current_clue_number.fill_(0)
        self.turn_count.fill_(0)

    def give_clue(self, clue_indices: torch.Tensor, clue_numbers: torch.Tensor) -> None:
        """
        Process clue-giving action for all games.

        Only active spymasters' clues are applied (based on masks).

        Args:
            clue_indices: [B] clue vocabulary indices
            clue_numbers: [B] clue numbers (1-N)
        """
        # Ensure tensors are on the correct device
        clue_indices = clue_indices.to(self.device)
        clue_numbers = clue_numbers.to(self.device)

        # Compute mask for active spymaster
        is_active = (~self.game_over) & (self.phase == self.SPYMASTER_PHASE)

        # Update state only for active games
        self.current_clue_index = torch.where(is_active, clue_indices, self.current_clue_index)
        self.current_clue_number = torch.where(is_active, clue_numbers, self.current_clue_number)
        self.remaining_guesses = torch.where(is_active, clue_numbers + 1, self.remaining_guesses)

        # Transition to guesser phase
        self.phase = torch.where(is_active, self.GUESSER_PHASE, self.phase)

    def guess(self, tile_indices: torch.Tensor) -> None:
        """
        Process guess action for all games using vectorized torch ops.

        Args:
            tile_indices: [B] tile indices to reveal
        """
        tile_indices = tile_indices.to(self.device).to(torch.int32).reshape(-1)
        if tile_indices.shape[0] != self.batch_size:
            raise ValueError(f"Expected tile_indices of length {self.batch_size}, got shape {tile_indices.shape}")

        is_active = (~self.game_over) & (self.phase == self.GUESSER_PHASE) & (self.remaining_guesses > 0)
        if not torch.any(is_active):
            return

        active_idx = torch.nonzero(is_active, as_tuple=False).squeeze(-1)
        chosen_tiles = tile_indices[active_idx]

        valid_tiles = (chosen_tiles >= 0) & (chosen_tiles < self.board_size)
        active_idx = active_idx[valid_tiles]
        chosen_tiles = chosen_tiles[valid_tiles]
        if active_idx.numel() == 0:
            return

        unrevealed_mask = ~self.revealed[active_idx, chosen_tiles]
        active_idx = active_idx[unrevealed_mask]
        chosen_tiles = chosen_tiles[unrevealed_mask]
        if active_idx.numel() == 0:
            return

        self.revealed[active_idx, chosen_tiles] = True
        self.remaining_guesses[active_idx] -= 1

        tile_colors = self.colors[active_idx, chosen_tiles]
        current_teams = self.current_team[active_idx].clone()

        assassin_mask = tile_colors == self.ASSASSIN
        if torch.any(assassin_mask):
            assassin_idx = active_idx[assassin_mask]
            self.game_over[assassin_idx] = True
            self.winner[assassin_idx] = 1 - current_teams[assassin_mask]

        team_hit_mask = tile_colors == current_teams

        team_tiles = (self.colors == self.current_team.unsqueeze(1))
        unrevealed = ~self.revealed
        team_remaining = torch.sum(team_tiles & unrevealed, dim=1)

        victory_mask = team_hit_mask & (team_remaining[active_idx] == 0)
        if torch.any(victory_mask):
            victory_idx = active_idx[victory_mask]
            self.game_over[victory_idx] = True
            self.winner[victory_idx] = current_teams[victory_mask]

        still_running = ~(assassin_mask | victory_mask)
        wrong_color_mask = (~team_hit_mask) & still_running
        no_guess_mask = (self.remaining_guesses[active_idx] == 0) & team_hit_mask & still_running

        turn_end_idx = active_idx[wrong_color_mask | no_guess_mask]
        self._end_turn_batch(turn_end_idx)

    def end_turn_early(self, mask: Optional[torch.Tensor] = None) -> None:
        """
        End turn early for specified games.

        Args:
            mask: [B] bool tensor indicating which games to end turn for.
                  If None, ends turn for all active guesser games.
        """
        if mask is None:
            mask = (~self.game_over) & (self.phase == self.GUESSER_PHASE)
        else:
            mask = mask.to(self.device).to(torch.bool)

        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        self._end_turn_batch(indices)

    def _end_turn(self, b: int) -> None:
        """
        End turn for game b and switch teams.

        Args:
            b: Batch index
        """
        self._end_turn_batch(torch.tensor([b], dtype=torch.int64, device=self.device))

    def _end_turn_batch(self, indices: torch.Tensor) -> None:
        """
        Vectorized helper to end turns for multiple games.

        Args:
            indices: Tensor of batch indices to end turns for
        """
        if indices.numel() == 0:
            return

        self.current_clue_index[indices] = -1
        self.current_clue_number[indices] = 0
        self.remaining_guesses[indices] = 0
        self.current_team[indices] = 1 - self.current_team[indices]
        self.phase[indices] = self.SPYMASTER_PHASE
        self.turn_count[indices] += 1

    def get_active_agent_masks(self) -> dict[str, torch.Tensor]:
        """
        Compute masks for which agent is active in each game.

        Returns:
            Dictionary mapping agent_id to [B] bool tensor:
                "red_spy": red spymaster is active
                "red_guess": red guesser is active
                "blue_spy": blue spymaster is active
                "blue_guess": blue guesser is active
        """
        is_spymaster = (self.phase == self.SPYMASTER_PHASE) & (~self.game_over)
        is_guesser = (self.phase == self.GUESSER_PHASE) & (~self.game_over)
        is_red = (self.current_team == self.RED)
        is_blue = (self.current_team == self.BLUE)

        return {
            "red_spy": is_spymaster & is_red,
            "red_guess": is_guesser & is_red,
            "blue_spy": is_spymaster & is_blue,
            "blue_guess": is_guesser & is_blue,
        }

    def get_unrevealed_counts(self) -> dict[str, torch.Tensor]:
        """
        Get counts of unrevealed tiles by color for each game.

        Returns:
            Dictionary mapping color name to [B] int tensor of counts
        """
        unrevealed = ~self.revealed

        return {
            "red": torch.sum((self.colors == self.RED) & unrevealed, dim=1),
            "blue": torch.sum((self.colors == self.BLUE) & unrevealed, dim=1),
            "neutral": torch.sum((self.colors == self.NEUTRAL) & unrevealed, dim=1),
            "assassin": torch.sum((self.colors == self.ASSASSIN) & unrevealed, dim=1),
        }

    def to(self, device: torch.device | str) -> GameState:
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
        self.colors = self.colors.to(device)
        self.revealed = self.revealed.to(device)
        self.current_team = self.current_team.to(device)
        self.phase = self.phase.to(device)
        self.game_over = self.game_over.to(device)
        self.winner = self.winner.to(device)
        self.remaining_guesses = self.remaining_guesses.to(device)
        self.current_clue_index = self.current_clue_index.to(device)
        self.current_clue_number = self.current_clue_number.to(device)
        self.turn_count = self.turn_count.to(device)

        return self
