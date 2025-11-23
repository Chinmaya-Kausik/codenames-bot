"""
Vector-based batched multi-agent Codenames environment.

This environment runs B parallel games with vector representations, exposing
a multi-agent API with fixed agent IDs for red/blue spymasters and guessers.
"""

from __future__ import annotations

from typing import Optional, Any
import torch

from core.game_state import GameState
from core.reality_layer import RealityLayer
from views.vector_view import VectorView


class VectorBatchEnv:
    """
    Batched, multi-agent, vector-based Codenames environment.

    Runs B parallel games with vector tile representations. Exposes a multi-agent
    API with fixed agent IDs: ["red_spy", "red_guess", "blue_spy", "blue_guess"].

    The environment:
    - Uses batched tensors for all state (B parallel games)
    - Only applies actions from the active agent in each game (based on masks)
    - Optionally uses a reality layer to snap continuous vectors to discrete vocab
    - Returns role-aware observations with team/role encodings

    Attributes:
        batch_size: Number of parallel games (B)
        board_size: Number of tiles per board (N)
        embedding_dim: Embedding dimension (D)
        device: Device tensors are stored on
        game_state: Core batched game state
        board_vectors: [B, N, D] tile embeddings for each game
        reality_layer: Optional reality layer for clue snapping
        agent_ids: List of fixed agent IDs
    """

    # Fixed agent IDs
    AGENT_IDS = ["red_spy", "red_guess", "blue_spy", "blue_guess"]

    def __init__(
        self,
        batch_size: int,
        board_size: int = 25,
        embedding_dim: int = 384,
        reality_layer: Optional[RealityLayer] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None
    ):
        """
        Initialize vector batch environment.

        Args:
            batch_size: Number of parallel games
            board_size: Number of tiles per board (default 25 for 5x5)
            embedding_dim: Embedding dimension for vectors
            reality_layer: Optional reality layer for clue snapping
            seed: Random seed for initialization
            device: Device to store tensors on (cpu/cuda/mps)
        """
        if device is None:
            from utils.device import get_device
            device = get_device()
        self.device = torch.device(device) if isinstance(device, str) else device

        self.batch_size = batch_size
        self.board_size = board_size
        self.embedding_dim = embedding_dim
        self.reality_layer = reality_layer

        # Initialize game state
        self.game_state = GameState(
            batch_size=batch_size,
            board_size=board_size,
            seed=seed,
            device=self.device
        )

        # Initialize board vectors (random for each game)
        self.board_vectors = VectorView.create_batched(
            batch_size=batch_size,
            board_size=board_size,
            embedding_dim=embedding_dim,
            seed=seed,
            normalize=True,
            device=self.device
        )

        # Store current clue vectors for observations
        self.current_clue_vectors = torch.zeros((batch_size, embedding_dim), device=self.device)

        self.agent_ids = self.AGENT_IDS.copy()

    def reset(self, seed: Optional[int] = None) -> dict[str, torch.Tensor]:
        """
        Reset all games to initial state.

        Args:
            seed: Random seed for reset

        Returns:
            obs_dict: Dictionary mapping agent_id to batched observations
        """
        # Reset game state
        self.game_state.reset(seed)

        # Regenerate board vectors
        self.board_vectors = VectorView.create_batched(
            batch_size=self.batch_size,
            board_size=self.board_size,
            embedding_dim=self.embedding_dim,
            seed=seed,
            normalize=True,
            device=self.device
        )

        # Clear clue vectors
        self.current_clue_vectors = torch.zeros((self.batch_size, self.embedding_dim), device=self.device)

        return self._get_observations()

    def step(
        self,
        actions_dict: dict[str, dict[str, torch.Tensor]]
    ) -> tuple[dict, dict, dict, dict]:
        """
        Execute one step for all games.

        Only actions from the active agent in each game are applied.
        All agents must provide actions, but inactive agents' actions are ignored.

        Args:
            actions_dict: Dictionary mapping agent_id to action dict containing:
                For spymasters: {"clue_vec": [B, D], "clue_number": [B]}
                For guessers: {"tile_index": [B]}

        Returns:
            Tuple of (obs_dict, rewards_dict, dones_dict, infos_dict), where:
                - obs_dict: observations for each agent
                - rewards_dict: rewards for each agent [B]
                - dones_dict: done flags for each agent [B]
                - infos_dict: additional info for each agent
        """
        # Get active agent masks
        active_masks = self.game_state.get_active_agent_masks()

        # Determine which phase we're in and process accordingly
        is_spymaster_phase = self.game_state.phase == GameState.SPYMASTER_PHASE
        is_guesser_phase = self.game_state.phase == GameState.GUESSER_PHASE

        rewards = {agent_id: torch.zeros(self.batch_size, device=self.device) for agent_id in self.agent_ids}

        # Process spymaster actions if in spymaster phase
        if torch.any(is_spymaster_phase & ~self.game_state.game_over):
            self._process_spymaster_actions(actions_dict, active_masks)

        # Process guesser actions if in guesser phase
        if torch.any(is_guesser_phase & ~self.game_state.game_over):
            rewards = self._process_guesser_actions(actions_dict, active_masks)

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

        Args:
            actions_dict: Actions from all agents
            active_masks: Masks indicating which agents are active
        """
        from envs.common import process_spymaster_actions

        # Use shared helper for common logic, passing previous clue data
        clue_indices, clue_numbers, final_vecs = process_spymaster_actions(
            batch_size=self.batch_size,
            device=self.device,
            actions_dict=actions_dict,
            active_masks=active_masks,
            reality_layer=self.reality_layer,
            clue_type="vector",
            embedding_dim=self.embedding_dim,
            prev_clue_indices=self.game_state.current_clue_index,
            prev_clue_numbers=self.game_state.current_clue_number,
            prev_clue_outputs=self.current_clue_vectors
        )

        # Store clue vectors for observations
        self.current_clue_vectors = final_vecs

        # Update game state with clues
        self.game_state.give_clue(clue_indices, clue_numbers)

    def _process_guesser_actions(
        self,
        actions_dict: dict,
        active_masks: dict
    ) -> dict:
        """
        Process guesser tile selection actions.

        Args:
            actions_dict: Actions from all agents
            active_masks: Masks indicating which agents are active

        Returns:
            Dictionary mapping agent_id to rewards [B]
        """
        # Combine actions from both guessers
        red_guess_actions = actions_dict.get("red_guess", {})
        blue_guess_actions = actions_dict.get("blue_guess", {})

        red_tile_indices = red_guess_actions.get("tile_index", torch.zeros(self.batch_size, dtype=torch.int32, device=self.device))
        blue_tile_indices = blue_guess_actions.get("tile_index", torch.zeros(self.batch_size, dtype=torch.int32, device=self.device))

        # Ensure tensors are on correct device
        red_tile_indices = red_tile_indices.to(self.device)
        blue_tile_indices = blue_tile_indices.to(self.device)

        # Select actions based on which guesser is active
        tile_indices = torch.where(
            active_masks["red_guess"],
            red_tile_indices,
            blue_tile_indices
        )

        # Store state before guess for reward calculation
        prev_team_counts = self.game_state.get_unrevealed_counts()
        prev_game_over = self.game_state.game_over.clone()

        # Apply guess
        self.game_state.guess(tile_indices)

        # Calculate rewards
        rewards = self._calculate_rewards(prev_team_counts, prev_game_over)

        return rewards

    def _calculate_rewards(self, prev_counts: dict, prev_game_over: torch.Tensor) -> dict:
        """
        Calculate rewards for each agent based on state changes.

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
                opponent_key = "blue"
            else:
                team_key = "blue"
                opponent_key = "red"

            # Reward structure:
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
            if "red" in agent_id:
                team_code = GameState.RED
            else:
                team_code = GameState.BLUE

            won_games = newly_finished & (self.game_state.winner == team_code)
            lost_games = newly_finished & (self.game_state.winner != team_code) & (self.game_state.winner >= 0)

            agent_rewards += 10 * won_games.to(torch.float32)
            agent_rewards -= 10 * lost_games.to(torch.float32)

            rewards[agent_id] = agent_rewards

        return rewards

    def _base_observation(self) -> dict[str, Any]:
        """
        Build shared observations common to all agents.

        Returns:
            Dictionary with cloned tensors to prevent mutation
        """
        base = {}

        # Common state tensors (cloned to prevent mutation)
        base["board_vectors"] = self.board_vectors.clone()
        base["revealed"] = self.game_state.revealed.clone()
        base["current_team"] = self.game_state.current_team.clone()
        base["phase"] = self.game_state.phase.clone()

        # Tensors for role-specific use (cloned to prevent mutation)
        base["colors"] = self.game_state.colors.clone()
        base["current_clue_vec"] = self.current_clue_vectors.clone()
        base["current_clue_number"] = self.game_state.current_clue_number.clone()
        base["remaining_guesses"] = self.game_state.remaining_guesses.clone()

        return base

    def _get_observations(self) -> dict[str, Any]:
        """
        Build role-aware observations for all agents.

        Returns:
            Dictionary mapping agent_id to observation dict
        """
        # Get base observation (cloned tensors)
        base = self._base_observation()
        obs_dict = {}

        for agent_id in self.agent_ids:
            # Determine team and role
            team = "red" if "red" in agent_id else "blue"
            is_spymaster = "spy" in agent_id

            # Build observation with cloned tensors from base
            obs = {
                "board_vectors": base["board_vectors"],
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

                # Guessers see current clue
                obs["current_clue_vec"] = base["current_clue_vec"]
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
        }

        # Add clue words if reality layer has tokens (computed once)
        if self.reality_layer is not None:
            clue_words = self.reality_layer.get_words(self.game_state.current_clue_index)
            if clue_words is not None:
                shared_info["clue_words"] = clue_words

        # All agents get the same info dict
        infos_dict = {}
        for agent_id in self.agent_ids:
            infos_dict[agent_id] = shared_info

        return infos_dict

    def to(self, device: torch.device | str) -> VectorBatchEnv:
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
        self.board_vectors = self.board_vectors.to(device)
        self.current_clue_vectors = self.current_clue_vectors.to(device)
        if self.reality_layer is not None:
            self.reality_layer.to(device)

        return self
