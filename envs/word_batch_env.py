"""
Word-based batched multi-agent Codenames environment.

This environment runs B parallel games with word representations, exposing
a multi-agent API with fixed agent IDs for red/blue spymasters and guessers.
"""

from __future__ import annotations

from typing import Optional, Any, Callable
import torch
import random

from core.game_state import GameState
from core.reality_layer import RealityLayer
from views.word_view import WordView
from utils.wordlist import WORD_POOL


def default_sparse_reward(
    prev_state: GameState,
    new_state: GameState,
    agent_id: str,
    team_idx: int
) -> float:
    """
    Default sparse reward: +1 for winning, -1 for losing, 0 otherwise.

    Args:
        prev_state: GameState before action
        new_state: GameState after action
        agent_id: Agent ID string
        team_idx: Team index for this agent

    Returns:
        Float reward
    """
    # Win reward
    if new_state.game_over.any() and (new_state.winner == team_idx).any():
        return 1.0
    # Loss reward
    elif new_state.game_over.any() and (new_state.winner != team_idx).any() and (new_state.winner >= 0).any():
        return -1.0
    # All other cases
    return 0.0


class WordBatchEnv:
    """
    Batched, multi-agent, word-based Codenames environment.

    Runs B parallel games with word representations. Exposes a multi-agent
    API with fixed agent IDs: ["red_spy", "red_guess", "blue_spy", "blue_guess"].

    The environment:
    - Uses batched tensors for all state (B parallel games)
    - Only applies actions from the active agent in each game (based on masks)
    - Optionally uses a reality layer for clue token lookup
    - Returns role-aware observations with team/role encodings
    - Supports word-based actions only (use VectorBatchEnv for vector-based clues)

    Attributes:
        batch_size: Number of parallel games (B)
        board_size: Number of tiles per board (N)
        device: Device tensors are stored on
        game_state: Core batched game state
        word_views: List of WordView instances, one per game
        reality_layer: Optional reality layer for token lookup
        reward_fn: Custom reward function (defaults to sparse win/loss)
        agent_ids: List of fixed agent IDs
    """

    # Fixed agent IDs
    AGENT_IDS = ["red_spy", "red_guess", "blue_spy", "blue_guess"]

    def __init__(
        self,
        batch_size: int,
        board_size: int = 25,
        reality_layer: Optional[RealityLayer] = None,
        word_pool: Optional[list[str]] = None,
        reward_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None
    ):
        """
        Initialize word batch environment.

        Args:
            batch_size: Number of parallel games
            board_size: Number of tiles per board (default 25 for 5x5)
            reality_layer: Optional reality layer for vector clue snapping
            word_pool: Optional word pool (defaults to WORD_POOL from utils)
            reward_fn: Optional custom reward function
            seed: Random seed for initialization
            device: Device to store tensors on (cpu/cuda/mps)
        """
        if device is None:
            from utils.device import get_device
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        self.batch_size = batch_size
        self.board_size = board_size
        self.reality_layer = reality_layer
        self.reward_fn = reward_fn or default_sparse_reward

        # Use provided word pool or default
        self.word_pool = word_pool if word_pool is not None else WORD_POOL

        # Initialize game state
        self.game_state = GameState(
            batch_size=batch_size,
            board_size=board_size,
            seed=seed,
            device=self.device
        )

        # Initialize word views (one per game in batch)
        rng = random.Random(seed)
        self.word_views = [
            WordView.create_random(
                board_size=board_size,
                word_pool=self.word_pool,
                seed=rng.randint(0, 2**31 - 1)
            )
            for _ in range(batch_size)
        ]

        # Store current clue words for observations
        self.current_clue_words = [""] * batch_size

        self.agent_ids = self.AGENT_IDS.copy()

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """
        Reset all games to initial state.

        Args:
            seed: Random seed for reset

        Returns:
            obs_dict: Dictionary mapping agent_id to batched observations
        """
        # Reset game state
        self.game_state.reset(seed)

        # Regenerate word views
        rng = random.Random(seed)
        self.word_views = [
            WordView.create_random(
                board_size=self.board_size,
                word_pool=self.word_pool,
                seed=rng.randint(0, 2**31 - 1)
            )
            for _ in range(self.batch_size)
        ]

        # Clear clue words
        self.current_clue_words = [""] * self.batch_size

        return self._get_observations()

    def step(
        self,
        actions_dict: dict[str, dict[str, Any]]
    ) -> tuple[dict, dict, dict, dict]:
        """
        Execute one step for all games.

        Only actions from the active agent in each game are applied.
        All agents must provide actions, but inactive agents' actions are ignored.

        Args:
            actions_dict: Dictionary mapping agent_id to action dict containing:
                For spymasters:
                    {"clue": [B] list of strings, "clue_number": [B] tensor}
                For guessers:
                    {"word": [B] list of strings}
                    OR {"tile_index": [B] tensor}

        Returns:
            Tuple of (obs_dict, rewards_dict, dones_dict, infos_dict)
        """
        # Get active agent masks
        active_masks = self.game_state.get_active_agent_masks()

        # Determine which phase we're in
        is_spymaster_phase = self.game_state.phase == GameState.SPYMASTER_PHASE
        is_guesser_phase = self.game_state.phase == GameState.GUESSER_PHASE

        # Store previous state for reward calculation
        prev_counts = self.game_state.get_unrevealed_counts()

        rewards = {agent_id: torch.zeros(self.batch_size, device=self.device) for agent_id in self.agent_ids}

        # Process spymaster actions if in spymaster phase
        if torch.any(is_spymaster_phase & ~self.game_state.game_over):
            self._process_spymaster_actions(actions_dict, active_masks)

        # Process guesser actions if in guesser phase
        if torch.any(is_guesser_phase & ~self.game_state.game_over):
            rewards = self._process_guesser_actions(actions_dict, active_masks, prev_counts)

        # Build outputs
        obs_dict = self._get_observations()
        rewards_dict = self._get_rewards(rewards, active_masks)
        dones_dict = self._get_dones()
        infos_dict = self._get_infos()

        return obs_dict, rewards_dict, dones_dict, infos_dict

    def _process_spymaster_actions(
        self,
        actions_dict: dict,
        active_masks: dict
    ) -> None:
        """
        Process spymaster clue-giving actions.

        Supports word-based clues only. Use VectorBatchEnv for vector-based clues.

        Args:
            actions_dict: Actions from all agents (expects "clue" and "clue_number")
            active_masks: Masks indicating which agents are active
        """
        from envs.common import process_spymaster_actions

        # Use shared helper for common logic, passing previous clue data
        clue_indices, clue_numbers, clue_words = process_spymaster_actions(
            batch_size=self.batch_size,
            device=self.device,
            actions_dict=actions_dict,
            active_masks=active_masks,
            reality_layer=self.reality_layer,
            clue_type="word",
            prev_clue_indices=self.game_state.current_clue_index,
            prev_clue_numbers=self.game_state.current_clue_number,
            prev_words=self.current_clue_words
        )

        # Store clue words for observations
        self.current_clue_words = clue_words

        # Update game state with clues
        self.game_state.give_clue(clue_indices, clue_numbers)

    def _word_to_index(self, b: int, word: str) -> int:
        """
        Convert a word guess to a tile index for a specific game.

        Args:
            b: Game index in batch
            word: Word string to convert (case-insensitive)

        Returns:
            Tile index

        Raises:
            ValueError: If word not found on the board
        """
        upper_word = word.upper()
        try:
            return self.word_views[b].get_index(upper_word)
        except ValueError:
            raise ValueError(f"Unknown guess '{word}' for game {b}")

    def _process_guesser_actions(
        self,
        actions_dict: dict,
        active_masks: dict,
        prev_counts: dict
    ) -> dict:
        """
        Process guesser tile selection actions.

        Supports both word-based guesses and index-based guesses.

        Args:
            actions_dict: Actions from all agents
            active_masks: Masks indicating which agents are active
            prev_counts: Unrevealed counts before action

        Returns:
            Dictionary mapping agent_id to rewards [B]
        """
        # Get actions from both guessers
        red_guess_actions = actions_dict.get("red_guess", {})
        blue_guess_actions = actions_dict.get("blue_guess", {})

        # Initialize with zeros
        red_indices = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        blue_indices = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)

        # Extract red indices (check word-based first, then index-based)
        if "word" in red_guess_actions:
            word_list = red_guess_actions["word"]
            if isinstance(word_list, list):
                for b in range(min(len(word_list), self.batch_size)):
                    word = word_list[b]
                    red_indices[b] = self._word_to_index(b, word)
        elif "tile_index" in red_guess_actions:
            idx_array = red_guess_actions["tile_index"]
            if isinstance(idx_array, torch.Tensor):
                red_indices = idx_array.to(self.device).to(torch.int32)
            else:
                red_indices = torch.tensor(idx_array, dtype=torch.int32, device=self.device)

        # Extract blue indices (check word-based first, then index-based)
        if "word" in blue_guess_actions:
            word_list = blue_guess_actions["word"]
            if isinstance(word_list, list):
                for b in range(min(len(word_list), self.batch_size)):
                    word = word_list[b]
                    blue_indices[b] = self._word_to_index(b, word)

        # Vectorized selection based on active masks
        tile_indices = torch.where(
            active_masks["red_guess"],
            red_indices,
            blue_indices
        )

        # Track game state before guess
        prev_game_over = self.game_state.game_over.clone()

        # Apply guess
        self.game_state.guess(tile_indices)

        # Calculate rewards
        rewards = self._calculate_rewards(prev_counts, prev_game_over)

        return rewards

    def _calculate_rewards(self, prev_counts: dict, prev_game_over: torch.Tensor) -> dict:
        """
        Calculate rewards for each agent based on state changes.

        Uses the configured reward_fn if provided, otherwise uses default sparse rewards.

        Args:
            prev_counts: Unrevealed counts before action
            prev_game_over: Game over status before action

        Returns:
            Dictionary mapping agent_id to rewards [B]
        """
        new_counts = self.game_state.get_unrevealed_counts()

        rewards = {}

        # Track which games just finished (not already finished)
        newly_finished = ~prev_game_over & self.game_state.game_over

        for agent_id in self.agent_ids:
            agent_rewards = torch.zeros(self.batch_size, device=self.device)

            # Determine team for this agent
            if "red" in agent_id:
                team_key = "red"
                team_idx = GameState.RED
                opponent_key = "blue"
            else:
                team_key = "blue"
                team_idx = GameState.BLUE
                opponent_key = "red"

            # Default reward structure (will be used if reward_fn not provided):
            # +1 for revealing own team tile
            # -1 for revealing opponent tile
            # -10 for revealing assassin
            # +10 for winning
            # -10 for losing

            # Team tile revealed
            team_revealed = prev_counts[team_key] - new_counts[team_key]
            agent_rewards += team_revealed.to(torch.float32)

            # Opponent tile revealed
            opp_revealed = prev_counts[opponent_key] - new_counts[opponent_key]
            agent_rewards -= opp_revealed.to(torch.float32)

            # Assassin revealed
            assassin_revealed = prev_counts["assassin"] - new_counts["assassin"]
            agent_rewards -= 10 * assassin_revealed.to(torch.float32)

            # Game end rewards (only for newly finished games)
            won_games = newly_finished & (self.game_state.winner == team_idx)
            lost_games = newly_finished & (self.game_state.winner != team_idx) & (self.game_state.winner >= 0)

            agent_rewards += 10 * won_games.to(torch.float32)
            agent_rewards -= 10 * lost_games.to(torch.float32)

            rewards[agent_id] = agent_rewards

        return rewards

    def _base_observation(self) -> dict[str, Any]:
        """
        Build shared observations common to all agents.

        Returns:
            Dictionary with cloned tensors and lists for mutable data
        """
        base = {}

        # Words for each game: list of lists (will be copied per agent)
        base["words"] = [view.get_all_words() for view in self.word_views]

        # Common state tensors (cloned to prevent mutation)
        base["revealed"] = self.game_state.revealed.clone()
        base["current_team"] = self.game_state.current_team.clone()
        base["phase"] = self.game_state.phase.clone()

        # Tensors for role-specific use (cloned to prevent mutation)
        base["colors"] = self.game_state.colors.clone()
        base["current_clue"] = self.current_clue_words  # Will be copied per agent
        base["current_clue_number"] = self.game_state.current_clue_number.clone()
        base["remaining_guesses"] = self.game_state.remaining_guesses.clone()

        return base

    def _get_observations(self) -> dict[str, Any]:
        """
        Build role-aware observations for all agents.

        Returns:
            Dictionary mapping agent_id to observation dict
        """
        # Get base observation (cloned tensors and lists)
        base = self._base_observation()
        obs_dict = {}

        for agent_id in self.agent_ids:
            # Determine team and role
            team = "red" if "red" in agent_id else "blue"
            is_spymaster = "spy" in agent_id

            # Build observation with cloned tensors and fresh list copies per agent
            obs = {
                "words": [list(words) for words in base["words"]],
                "revealed": base["revealed"],
                "current_team": base["current_team"],
                "phase": base["phase"],
            }

            if is_spymaster:
                # Spymasters see colors
                obs["colors"] = base["colors"]
            else:
                # Guessers don't see colors
                obs["colors"] = None

                # Guessers see current clue (fresh copy)
                obs["current_clue"] = list(base["current_clue"])
                obs["current_clue_number"] = base["current_clue_number"]
                obs["remaining_guesses"] = base["remaining_guesses"]

            # Role and team encoding: [B, 4] one-hot (small, agent-specific)
            # [is_red, is_blue, is_spymaster, is_guesser]
            role_encoding = torch.zeros((self.batch_size, 4), device=self.device)
            role_encoding[:, 0] = 1 if team == "red" else 0
            role_encoding[:, 1] = 1 if team == "blue" else 0
            role_encoding[:, 2] = 1 if is_spymaster else 0
            role_encoding[:, 3] = 0 if is_spymaster else 1

            obs["role_encoding"] = role_encoding

            obs_dict[agent_id] = obs

        return obs_dict

    def _get_rewards(
        self,
        step_rewards: dict,
        active_masks: dict
    ) -> dict[str, torch.Tensor]:
        """
        Build rewards dict for all agents.

        Rewards are shared per team - both spymaster and guesser on the same team
        receive the same reward signal regardless of which role acted.

        Args:
            step_rewards: Rewards from this step
            active_masks: Active agent masks (not used for masking, kept for API compatibility)

        Returns:
            Dictionary mapping agent_id to rewards [B]
        """
        rewards_dict = {}

        for agent_id in self.agent_ids:
            # Return raw rewards - both roles on same team get same signal
            rewards_dict[agent_id] = step_rewards.get(agent_id, torch.zeros(self.batch_size, device=self.device))

        return rewards_dict

    def _get_dones(self) -> dict[str, torch.Tensor]:
        """
        Build dones dict.

        Returns:
            Dictionary mapping agent_id to done flags [B]
        """
        dones_dict = {}

        for agent_id in self.agent_ids:
            dones_dict[agent_id] = self.game_state.game_over.clone()

        return dones_dict

    def _get_infos(self) -> dict[str, dict]:
        """
        Build infos dict with additional information.

        Computes shared information once and reuses it across all agents.

        Returns:
            Dictionary mapping agent_id to info dict
        """
        # Build shared info once (expensive operations done once, tensors cloned)
        shared_info = {
            "winner": self.game_state.winner.clone(),
            "turn_count": self.game_state.turn_count.clone(),
            "unrevealed_counts": self.game_state.get_unrevealed_counts(),  # Computed once
            "clue_words": self.current_clue_words.copy(),
        }

        # Add revealed words for each game (computed once using tensor masks)
        revealed_words = [
            [
                self.word_views[b].get_word(i)
                for i in torch.nonzero(self.game_state.revealed[b], as_tuple=False).squeeze(-1).tolist()
            ]
            for b in range(self.batch_size)
        ]
        shared_info["revealed_words"] = revealed_words

        # All agents get the same info dict
        infos_dict = {}
        for agent_id in self.agent_ids:
            infos_dict[agent_id] = shared_info

        return infos_dict

    def to(self, device: torch.device | str) -> WordBatchEnv:
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
        self.game_state.to(device)
        if self.reality_layer is not None:
            self.reality_layer.to(device)

        return self
