"""
Multi-agent experiment runner for Codenames.

This module provides the MultiAgentCodenamesExperiment class for running
experiments with multi-agent environments and policies.
"""

from __future__ import annotations

from typing import Callable, Union, Optional, Any
import random
import torch

from envs.vector_batch_env import VectorBatchEnv
from envs.word_batch_env import WordBatchEnv
from experiments.trackers import GameTracker, SummaryTracker


class MultiAgentCodenamesExperiment:
    """
    Experiment runner for multi-agent Codenames environments.

    Runs games with specified policies and tracks results using GameTracker callbacks.
    Supports both VectorBatchEnv and WordBatchEnv.

    Example:
        ```python
        # Create experiment
        exp = MultiAgentCodenamesExperiment(
            env_factory=lambda seed: WordBatchEnv(batch_size=32, seed=seed),
            max_turns=50
        )

        # Define policies
        policy_map = {
            "red_spy": lambda obs: red_spymaster.get_clue(obs),
            "red_guess": lambda obs: red_guesser.get_guess(obs),
            "blue_spy": lambda obs: blue_spymaster.get_clue(obs),
            "blue_guess": lambda obs: blue_guesser.get_guess(obs),
        }

        # Run with summary tracker
        results = exp.run_games(
            policy_map=policy_map,
            n_games=100,
            tracker=SummaryTracker()
        )

        print(f"Red win rate: {results['red_win_rate']:.2f}")
        ```

    Attributes:
        env_factory: Callable that creates environment given seed
        max_turns: Maximum turns per game before timeout
    """

    def __init__(
        self,
        env_factory: Callable[[int], Union[VectorBatchEnv, WordBatchEnv]],
        max_turns: int = 50
    ):
        """
        Initialize experiment runner.

        Args:
            env_factory: Function that creates environment given seed.
                        Should return VectorBatchEnv or WordBatchEnv.
                        Example: lambda seed: WordBatchEnv(batch_size=32, seed=seed)
            max_turns: Maximum number of turns per game
        """
        self.env_factory = env_factory
        self.max_turns = max_turns

    def _slice_dict(self, data_dict: dict, n: int, keys_to_slice: Optional[set] = None) -> dict:
        """
        Slice each value in dict to first n elements along batch dimension.

        Recursively handles nested dictionaries.

        Args:
            data_dict: Dictionary with tensor/array/list values
            n: Number of elements to keep
            keys_to_slice: Optional set of keys to slice. If None, slice all keys.
                          If provided, only slice entries whose keys are in the set.

        Returns:
            Dictionary with sliced values
        """
        sliced = {}
        for key, value in data_dict.items():
            # Check if we should slice this key
            should_slice = keys_to_slice is None or key in keys_to_slice

            if not should_slice:
                # Don't slice this key, just copy it
                sliced[key] = value
            elif isinstance(value, dict):
                # Recursively slice nested dicts
                sliced[key] = self._slice_dict(value, n, keys_to_slice)
            elif isinstance(value, torch.Tensor):
                sliced[key] = value[:n]
            elif isinstance(value, list):
                sliced[key] = value[:n]
            elif value is None:
                sliced[key] = None
            else:
                # Scalar or other non-sliceable value
                sliced[key] = value
        return sliced

    def run_games(
        self,
        policy_map: dict[str, Callable[[dict], dict]],
        n_games: int,
        tracker: Optional[GameTracker] = None,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Any:
        """
        Run n_games with given policies and tracker.

        Args:
            policy_map: Dictionary mapping agent_id to policy function.
                       Policy functions take observation dict and return action dict.
                       Must include all agents in environment.
            n_games: Number of games to run
            tracker: GameTracker instance to collect data. If None, uses SummaryTracker.
            seed: Random seed for first game (incremented for subsequent games)
            verbose: If True, print progress

        Returns:
            Results from tracker.get_results()

        Raises:
            ValueError: If policy_map doesn't cover all agents
        """
        if tracker is None:
            tracker = SummaryTracker()

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # Run games
        games_completed = 0

        while games_completed < n_games:
            # Create environment
            game_seed = seed + games_completed
            env = self.env_factory(game_seed)

            # Validate policy_map
            missing_agents = set(env.agent_ids) - set(policy_map.keys())
            if missing_agents:
                raise ValueError(
                    f"policy_map missing policies for agents: {missing_agents}"
                )

            # Determine how many games this batch will complete
            # (environment might have batch_size > 1)
            batch_size = env.batch_size
            games_this_batch = min(batch_size, n_games - games_completed)

            # Reset environment
            obs_dict = env.reset(seed=game_seed)

            # Initialize sliced_infos in case loop doesn't run
            # Use empty infos since we don't have any step data yet
            sliced_infos = self._slice_dict(
                {agent_id: {} for agent_id in env.agent_ids},
                games_this_batch
            )

            # Run game loop
            for turn in range(self.max_turns):
                # Check if all games in batch are done (only check active games)
                if torch.all(env.game_state.game_over[:games_this_batch]):
                    break

                # Get actions from all policies
                actions_dict = {}
                for agent_id, policy in policy_map.items():
                    actions_dict[agent_id] = policy(obs_dict[agent_id])

                # Step environment
                obs_dict, rewards_dict, dones_dict, infos_dict = env.step(actions_dict)

                # Slice only env outputs (not actions_dict) to reduce overhead
                # Trackers rarely use actions, so we don't slice them
                sliced_obs = self._slice_dict(obs_dict, games_this_batch)
                sliced_rewards = self._slice_dict(rewards_dict, games_this_batch)
                sliced_dones = self._slice_dict(dones_dict, games_this_batch)
                sliced_infos = self._slice_dict(infos_dict, games_this_batch)

                # Call tracker callback with sliced data
                tracker.on_step(
                    step=turn,
                    obs_dict=sliced_obs,
                    actions_dict=actions_dict,  # Pass unsliced actions_dict
                    rewards_dict=sliced_rewards,
                    dones_dict=sliced_dones,
                    infos_dict=sliced_infos
                )

            # Episode end callback with sliced data
            tracker.on_episode_end(
                episode_idx=games_completed,
                final_infos=sliced_infos
            )

            games_completed += games_this_batch

            if verbose:
                print(f"Completed {games_completed}/{n_games} games")

        # Return results
        return tracker.get_results()

    def run_sweep(
        self,
        policy_factory: Callable[[dict], dict[str, Callable]],
        param_grid: list[dict],
        n_games_per_config: int = 10,
        tracker_factory: Optional[Callable[[], GameTracker]] = None,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> list[dict]:
        """
        Run parameter sweep over multiple configurations.

        Args:
            policy_factory: Function that takes params dict and returns policy_map
            param_grid: List of parameter dictionaries to try
            n_games_per_config: Number of games to run per configuration
            tracker_factory: Function that creates fresh tracker for each config.
                           If None, uses SummaryTracker.
            seed: Base random seed
            verbose: If True, print progress

        Returns:
            List of dicts, one per configuration, containing:
                - params: The parameter dict
                - results: Results from tracker
                - seed: Seed used for this configuration

        Example:
            ```python
            def make_policies(params):
                spy = EmbeddingSpymaster(
                    team="red",
                    params=SpymasterParams(**params)
                )
                return {
                    "red_spy": lambda obs: spy.get_clue(obs),
                    # ... other agents
                }

            param_grid = [
                {"n_candidate_clues": 10, "risk_tolerance": 1.0},
                {"n_candidate_clues": 50, "risk_tolerance": 2.0},
                {"n_candidate_clues": 100, "risk_tolerance": 3.0},
            ]

            results = exp.run_sweep(
                policy_factory=make_policies,
                param_grid=param_grid,
                n_games_per_config=100
            )
            ```
        """
        if tracker_factory is None:
            tracker_factory = lambda: SummaryTracker()

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        sweep_results = []

        for i, params in enumerate(param_grid):
            if verbose:
                print(f"\nConfiguration {i+1}/{len(param_grid)}: {params}")

            # Create policies for this configuration
            policy_map = policy_factory(params)

            # Create fresh tracker
            tracker = tracker_factory()

            # Run games with this configuration
            config_seed = seed + i * 10000  # Ensure different seeds per config
            results = self.run_games(
                policy_map=policy_map,
                n_games=n_games_per_config,
                tracker=tracker,
                seed=config_seed,
                verbose=verbose
            )

            # Store results
            sweep_results.append({
                "params": params,
                "results": results,
                "seed": config_seed
            })

        return sweep_results
