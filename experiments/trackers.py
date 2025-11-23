"""
Game trackers for multi-agent Codenames experiments.

Trackers receive callbacks during game execution and accumulate data
for analysis, training, or evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np
import torch


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class BaseBatchTracker(ABC):
    """
    Base class for batch-aware game trackers.

    Provides common utilities for handling batched data:
    - Agent ID storage on first step
    - Tensor-to-numpy conversion
    - Iteration over batched winners/turn counts

    Subclasses implement the actual tracking logic.
    """

    def __init__(self):
        """Initialize base tracker."""
        self._initialized = False
        self.agent_ids = []

    def _ensure_initialized(self, rewards_dict: dict[str, np.ndarray]) -> None:
        """
        Initialize agent IDs on first step.

        Args:
            rewards_dict: Rewards dictionary from first step
        """
        if not self._initialized:
            self.agent_ids = list(rewards_dict.keys())
            self._initialized = True

    def _iter_batch_infos(self, final_infos: dict[str, Any]):
        """
        Iterate over each game in a batched info dict.

        Yields tuples of (winner, turn_count) for each game in the batch.

        Args:
            final_infos: Info dict from environment (may be batched)

        Yields:
            Tuple of (winner, turn_count) for each game
        """
        # Get info from first agent (all have same game-level info)
        info = final_infos[self.agent_ids[0]]

        # Convert torch tensors to numpy
        winner_arr = _to_numpy(info['winner'])
        turn_count_arr = _to_numpy(info['turn_count'])

        # Handle batched data
        if hasattr(winner_arr, '__iter__') and len(winner_arr) > 0:
            batch_size = len(winner_arr)
            for b in range(batch_size):
                yield int(winner_arr[b]), int(turn_count_arr[b])
        else:
            # Single game
            winner = int(winner_arr) if np.ndim(winner_arr) == 0 else int(winner_arr[0])
            turns = int(turn_count_arr) if np.ndim(turn_count_arr) == 0 else int(turn_count_arr[0])
            yield winner, turns


class GameTracker(ABC):
    """
    Abstract base class for game trackers.

    Trackers receive callbacks during game execution:
    - on_step: Called after each environment step
    - on_episode_end: Called when a game ends
    - get_results: Returns accumulated results

    Subclasses implement these methods to collect different types of data.
    """

    @abstractmethod
    def on_step(
        self,
        step: int,
        obs_dict: dict[str, Any],
        actions_dict: dict[str, Any],
        rewards_dict: dict[str, np.ndarray],
        dones_dict: dict[str, np.ndarray],
        infos_dict: dict[str, Any]
    ) -> None:
        """
        Called after each environment step.

        Args:
            step: Current step number
            obs_dict: Observations for each agent (batched)
            actions_dict: Actions from each agent (batched)
            rewards_dict: Rewards for each agent [B] arrays
            dones_dict: Done flags for each agent [B] arrays
            infos_dict: Info dicts for each agent
        """
        pass

    @abstractmethod
    def on_episode_end(
        self,
        episode_idx: int,
        final_infos: dict[str, Any]
    ) -> None:
        """
        Called when an episode (game) ends.

        Args:
            episode_idx: Index of the completed episode
            final_infos: Final info dict from the environment
        """
        pass

    @abstractmethod
    def get_results(self) -> Any:
        """
        Get accumulated results.

        Returns:
            Results in tracker-specific format
        """
        pass

    def reset(self) -> None:
        """
        Reset tracker state (optional).

        Default implementation does nothing. Override if tracker needs reset.
        """
        pass


class SummaryTracker(BaseBatchTracker, GameTracker):
    """
    Tracker that accumulates summary statistics.

    Computes running statistics across all games:
    - Win rates per team
    - Average rewards per agent
    - Average game length
    - Total games played

    Memory efficient - only stores aggregated statistics, not individual games.
    """

    def __init__(self):
        """Initialize summary tracker."""
        super().__init__()
        self.total_games = 0
        self.total_steps = 0

        # Per-agent statistics (accumulated per episode)
        self.total_rewards = {}
        self.reward_sum_sq = {}  # For computing variance

        # Win statistics
        self.red_wins = 0
        self.blue_wins = 0

        # Game length statistics
        self.total_turns = 0

        # Current episode accumulation
        self.current_episode_rewards = {}

    def on_step(
        self,
        step: int,
        obs_dict: dict[str, Any],
        actions_dict: dict[str, Any],
        rewards_dict: dict[str, np.ndarray],
        dones_dict: dict[str, np.ndarray],
        infos_dict: dict[str, Any]
    ) -> None:
        """Accumulate rewards for current episode."""
        # Initialize on first step
        if not self._initialized:
            self._ensure_initialized(rewards_dict)
            for agent_id in self.agent_ids:
                self.total_rewards[agent_id] = 0.0
                self.reward_sum_sq[agent_id] = 0.0
                self.current_episode_rewards[agent_id] = 0.0

        # Accumulate rewards for current episode (convert torch tensors to numpy if needed)
        for agent_id, rewards in rewards_dict.items():
            rewards_np = _to_numpy(rewards)
            self.current_episode_rewards[agent_id] += np.sum(rewards_np)

        self.total_steps += 1

    def on_episode_end(
        self,
        episode_idx: int,
        final_infos: dict[str, Any]
    ) -> None:
        """Update episode statistics."""
        # Update per-episode reward statistics (before incrementing games)
        for agent_id in self.agent_ids:
            episode_total = self.current_episode_rewards[agent_id]
            self.total_rewards[agent_id] += episode_total
            self.reward_sum_sq[agent_id] += episode_total ** 2

        # Iterate over each game in batch using base class utility
        for winner, turns in self._iter_batch_infos(final_infos):
            if winner == 0:  # Red wins
                self.red_wins += 1
            elif winner == 1:  # Blue wins
                self.blue_wins += 1

            self.total_turns += turns
            self.total_games += 1

        # Reset for next episode
        for agent_id in self.agent_ids:
            self.current_episode_rewards[agent_id] = 0.0

    def get_results(self) -> dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with:
                - total_games: Number of games played
                - red_win_rate: Fraction of games won by red
                - blue_win_rate: Fraction of games won by blue
                - avg_turns: Average game length
                - rewards_per_agent: Dict of average rewards per agent
                - reward_std_per_agent: Dict of reward std per agent
        """
        if self.total_games == 0:
            return {
                "total_games": 0,
                "red_win_rate": 0.0,
                "blue_win_rate": 0.0,
                "avg_turns": 0.0,
                "rewards_per_agent": {},
                "reward_std_per_agent": {},
            }

        # Compute average rewards and standard deviations
        avg_rewards = {}
        std_rewards = {}

        for agent_id in self.agent_ids:
            mean = self.total_rewards[agent_id] / self.total_games
            # Var = E[X^2] - E[X]^2
            variance = (self.reward_sum_sq[agent_id] / self.total_games) - (mean ** 2)
            variance = max(0, variance)  # Handle numerical errors

            avg_rewards[agent_id] = mean
            std_rewards[agent_id] = np.sqrt(variance)

        return {
            "total_games": self.total_games,
            "red_win_rate": self.red_wins / self.total_games,
            "blue_win_rate": self.blue_wins / self.total_games,
            "avg_turns": self.total_turns / self.total_games,
            "rewards_per_agent": avg_rewards,
            "reward_std_per_agent": std_rewards,
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.__init__()


class EpisodeTracker(BaseBatchTracker, GameTracker):
    """
    Tracker that stores per-episode results.

    Stores summary information for each completed game:
    - Total rewards per agent
    - Winner
    - Number of turns
    - Episode index

    More memory intensive than SummaryTracker but less than TrajectoryTracker.
    Useful for analyzing individual games or parameter sweeps.
    """

    def __init__(self):
        """Initialize episode tracker."""
        super().__init__()
        self.episodes = []
        self.current_episode_rewards = {}

    def on_step(
        self,
        step: int,
        obs_dict: dict[str, Any],
        actions_dict: dict[str, Any],
        rewards_dict: dict[str, np.ndarray],
        dones_dict: dict[str, np.ndarray],
        infos_dict: dict[str, Any]
    ) -> None:
        """Accumulate rewards for current episode(s)."""
        # Initialize on first step using base class utility
        if not self._initialized:
            self._ensure_initialized(rewards_dict)
            # Determine batch size from rewards
            first_reward = next(iter(rewards_dict.values()))
            first_reward_np = _to_numpy(first_reward)
            batch_size = len(first_reward_np) if hasattr(first_reward_np, '__len__') else 1

            # Initialize per-game accumulators
            self.current_episode_rewards = {
                agent_id: np.zeros(batch_size) for agent_id in self.agent_ids
            }

        # Accumulate rewards per game in batch
        for agent_id, rewards in rewards_dict.items():
            rewards_np = _to_numpy(rewards)
            if hasattr(rewards_np, '__len__'):
                self.current_episode_rewards[agent_id] += rewards_np
            else:
                self.current_episode_rewards[agent_id] += rewards_np

    def on_episode_end(
        self,
        episode_idx: int,
        final_infos: dict[str, Any]
    ) -> None:
        """Store episode results and reset for next episode(s)."""
        # Iterate over each game in batch using base class utility
        for idx, (winner, turns) in enumerate(self._iter_batch_infos(final_infos)):
            # Extract rewards for this game
            game_rewards = {}
            for agent_id in self.agent_ids:
                reward_arr = self.current_episode_rewards[agent_id]
                if hasattr(reward_arr, '__len__'):
                    game_rewards[agent_id] = float(reward_arr[idx])
                else:
                    game_rewards[agent_id] = float(reward_arr)

            # Store episode data
            episode_data = {
                "episode_idx": episode_idx + idx,
                "total_rewards": game_rewards,
                "winner": winner,
                "turns": turns,
            }

            self.episodes.append(episode_data)

        # Reset for next episode
        batch_size = len(self.current_episode_rewards[self.agent_ids[0]])
        for agent_id in self.agent_ids:
            self.current_episode_rewards[agent_id] = np.zeros(batch_size)

    def get_results(self) -> list[dict]:
        """
        Get list of episode results.

        Returns:
            List of dicts, one per episode, containing:
                - episode_idx: Episode index
                - total_rewards: Dict of total rewards per agent
                - winner: Winning team (0=red, 1=blue, -1=none)
                - turns: Number of turns in game
        """
        return self.episodes

    def reset(self) -> None:
        """Clear all episode data."""
        self.__init__()


class TrajectoryTracker(BaseBatchTracker, GameTracker):
    """
    Tracker that stores full step-by-step trajectories.

    Stores complete episode data including observations, actions, and rewards
    at each timestep. Most memory intensive but provides complete information
    for debugging, visualization, or detailed analysis.

    Useful for:
    - Training RL agents (need step-by-step data)
    - Debugging agent behavior
    - Creating replay visualizations
    """

    def __init__(self, store_observations: bool = True, store_actions: bool = True):
        """
        Initialize trajectory tracker.

        Args:
            store_observations: Whether to store observations (can be large)
            store_actions: Whether to store actions
        """
        super().__init__()
        self.store_observations = store_observations
        self.store_actions = store_actions

        self.episodes = []
        self.current_trajectory = []

    def on_step(
        self,
        step: int,
        obs_dict: dict[str, Any],
        actions_dict: dict[str, Any],
        rewards_dict: dict[str, np.ndarray],
        dones_dict: dict[str, np.ndarray],
        infos_dict: dict[str, Any]
    ) -> None:
        """Store step data in current trajectory."""
        # Initialize on first step using base class utility
        if not self._initialized:
            self._ensure_initialized(rewards_dict)

        # Build step data (convert torch tensors to numpy for storage)
        step_data = {
            "step": step,
            "rewards": {aid: _to_numpy(rewards_dict[aid]) for aid in self.agent_ids},
            "dones": {aid: _to_numpy(dones_dict[aid]) for aid in self.agent_ids},
        }

        if self.store_observations:
            # Store observations (careful - can be large)
            step_data["observations"] = {
                aid: {k: _to_numpy(v) if isinstance(v, (np.ndarray, torch.Tensor)) else v
                      for k, v in obs_dict[aid].items()}
                for aid in self.agent_ids
            }

        if self.store_actions:
            step_data["actions"] = {
                aid: {k: _to_numpy(v) if isinstance(v, (np.ndarray, torch.Tensor)) else v
                      for k, v in actions_dict[aid].items()}
                for aid in self.agent_ids
            }

        self.current_trajectory.append(step_data)

    def on_episode_end(
        self,
        episode_idx: int,
        final_infos: dict[str, Any]
    ) -> None:
        """Store complete trajectory and reset for next episode."""
        # Use base class utility to get winner and turns (takes first from batch)
        for winner, turns in self._iter_batch_infos(final_infos):
            # Store episode with trajectory (only first game if batched)
            episode_data = {
                "episode_idx": episode_idx,
                "trajectory": self.current_trajectory,
                "winner": winner,
                "turns": turns,
                "total_rewards": self._compute_total_rewards(),
            }

            self.episodes.append(episode_data)
            break  # Only store first game from batch

        # Reset for next episode
        self.current_trajectory = []

    def _compute_total_rewards(self) -> dict[str, float]:
        """Compute total rewards from trajectory."""
        total_rewards = {aid: 0.0 for aid in self.agent_ids}

        for step_data in self.current_trajectory:
            for aid in self.agent_ids:
                total_rewards[aid] += np.sum(step_data["rewards"][aid])

        return total_rewards

    def get_results(self) -> list[dict]:
        """
        Get list of episodes with full trajectories.

        Returns:
            List of dicts, one per episode, containing:
                - episode_idx: Episode index
                - trajectory: List of step data (obs, actions, rewards, dones)
                - winner: Winning team
                - turns: Number of turns
                - total_rewards: Dict of total rewards per agent
        """
        return self.episodes

    def reset(self) -> None:
        """Clear all trajectory data."""
        self.__init__(
            store_observations=self.store_observations,
            store_actions=self.store_actions
        )
