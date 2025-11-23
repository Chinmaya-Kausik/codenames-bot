"""
Single-agent wrapper for multi-agent Codenames environments.

This wrapper allows training a single agent while auto-controlling other agents
with fixed policies, exposing a simplified single-agent API.
"""

from __future__ import annotations

from typing import Union, Callable, Any
import numpy as np

from envs.vector_batch_env import VectorBatchEnv
from envs.word_batch_env import WordBatchEnv


class SingleAgentWrapper:
    """
    Wrapper for training a single agent in a multi-agent environment.

    Internally uses the multi-agent environment but exposes a single-agent API:
    - reset() → obs (for focused agent)
    - step(action) → (obs, reward, done, info)

    Other agents are automatically controlled using the provided policy_map.

    Example:
        ```python
        env = WordBatchEnv(batch_size=4)

        # Define policies for other agents
        policy_map = {
            "red_spy": lambda obs: random_spymaster.get_clue(obs),
            "blue_spy": lambda obs: random_spymaster.get_clue(obs),
            "blue_guess": lambda obs: random_guesser.get_guess(obs),
        }

        # Wrap environment to focus on red_guess agent
        wrapped_env = SingleAgentWrapper(
            env=env,
            agent_id="red_guess",
            policy_map=policy_map
        )

        # Use standard single-agent API
        obs = wrapped_env.reset()
        obs, reward, done, info = wrapped_env.step(action)
        ```

    Attributes:
        env: The underlying multi-agent environment
        agent_id: ID of the agent to focus on
        policy_map: Dictionary mapping agent_id to policy function
        batch_size: Number of parallel games
    """

    def __init__(
        self,
        env: Union[VectorBatchEnv, WordBatchEnv],
        agent_id: str,
        policy_map: dict[str, Callable[[dict], dict]]
    ):
        """
        Initialize single-agent wrapper.

        Args:
            env: Multi-agent environment (VectorBatchEnv or WordBatchEnv)
            agent_id: ID of the agent to focus on (e.g., "red_spy", "red_guess")
            policy_map: Dictionary mapping agent_id to policy function.
                       Policy functions take agent observation dict and return
                       action dict. Should include all agents EXCEPT agent_id.

        Raises:
            ValueError: If agent_id not in environment's agent list
            ValueError: If policy_map doesn't cover other agents
        """
        self.env = env
        self.agent_id = agent_id
        self.policy_map = policy_map
        self.batch_size = env.batch_size

        # Validate agent_id
        if agent_id not in env.agent_ids:
            raise ValueError(
                f"agent_id '{agent_id}' not in environment's agent IDs: {env.agent_ids}"
            )

        # Validate policy_map covers other agents
        other_agents = set(env.agent_ids) - {agent_id}
        missing_agents = other_agents - set(policy_map.keys())
        if missing_agents:
            raise ValueError(
                f"policy_map missing policies for agents: {missing_agents}"
            )

        # Store current observation dict for policy queries
        self._current_obs_dict = None

    def reset(self, seed: int = None) -> Any:
        """
        Reset the environment and return observation for focused agent.

        Args:
            seed: Random seed for reset

        Returns:
            Observation for the focused agent
        """
        self._current_obs_dict = self.env.reset(seed=seed)
        return self._current_obs_dict[self.agent_id]

    def step(self, action: dict) -> tuple[Any, np.ndarray, np.ndarray, dict]:
        """
        Step the environment with the focused agent's action.

        Other agents' actions are automatically generated using policy_map.

        Args:
            action: Action dict for the focused agent

        Returns:
            Tuple of (obs, reward, done, info) for the focused agent:
                - obs: Observation for focused agent
                - reward: Reward array [B] for focused agent
                - done: Done flags [B] for focused agent
                - info: Info dict for focused agent
        """
        # Build complete actions_dict
        actions_dict = {}

        for aid in self.env.agent_ids:
            if aid == self.agent_id:
                # Use provided action for focused agent
                actions_dict[aid] = action
            else:
                # Query policy for this agent using current observations
                policy = self.policy_map[aid]
                actions_dict[aid] = policy(self._current_obs_dict[aid])

        # Step the multi-agent environment
        obs_dict, rewards_dict, dones_dict, infos_dict = self.env.step(actions_dict)

        # Store updated observations for next step
        self._current_obs_dict = obs_dict

        # Return focused agent's view
        return (
            obs_dict[self.agent_id],
            rewards_dict[self.agent_id],
            dones_dict[self.agent_id],
            infos_dict[self.agent_id]
        )

    def close(self) -> None:
        """Close the environment (if applicable)."""
        if hasattr(self.env, 'close'):
            self.env.close()

    @property
    def agent_ids(self) -> list[str]:
        """Get list of all agent IDs in the underlying environment."""
        return self.env.agent_ids

    @property
    def game_state(self):
        """Access the underlying game state."""
        return self.env.game_state
