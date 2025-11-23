"""
Tests for SingleAgentWrapper.
"""

import torch
import pytest

from envs.word_batch_env import WordBatchEnv
from envs.vector_batch_env import VectorBatchEnv
from envs.wrappers import SingleAgentWrapper
from agents.spymaster import RandomSpymaster
from agents.guesser import RandomGuesser


def test_single_agent_wrapper_initialization():
    """Test SingleAgentWrapper initialization."""
    env = WordBatchEnv(batch_size=2, seed=42)

    # Create policies for other agents
    red_spy = RandomSpymaster(team="red", params=None)
    blue_spy = RandomSpymaster(team="blue", params=None)
    blue_guess = RandomGuesser(team="blue", params=None)

    policy_map = {
        "red_spy": lambda obs: red_spy.get_clue(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
        "blue_guess": lambda obs: blue_guess.get_guess(obs),
    }

    # Wrap environment to focus on red_guess
    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="red_guess",
        policy_map=policy_map
    )

    assert wrapped.agent_id == "red_guess"
    assert wrapped.batch_size == 2
    assert wrapped.env is env


def test_invalid_agent_id():
    """Test that invalid agent_id raises ValueError."""
    env = WordBatchEnv(batch_size=1, seed=42)

    policy_map = {
        "red_spy": lambda obs: {},
        "red_guess": lambda obs: {},
        "blue_spy": lambda obs: {},
        "blue_guess": lambda obs: {},
    }

    with pytest.raises(ValueError, match="not in environment's agent IDs"):
        SingleAgentWrapper(
            env=env,
            agent_id="invalid_agent",
            policy_map=policy_map
        )


def test_missing_policy_in_map():
    """Test that missing policy raises ValueError."""
    env = WordBatchEnv(batch_size=1, seed=42)

    # Missing blue_guess policy
    policy_map = {
        "red_spy": lambda obs: {},
        "blue_spy": lambda obs: {},
    }

    with pytest.raises(ValueError, match="policy_map missing policies"):
        SingleAgentWrapper(
            env=env,
            agent_id="red_guess",
            policy_map=policy_map
        )


def test_reset():
    """Test reset returns observation for focused agent."""
    env = WordBatchEnv(batch_size=2, seed=42)

    red_spy = RandomSpymaster(team="red", params=None)
    blue_spy = RandomSpymaster(team="blue", params=None)
    blue_guess = RandomGuesser(team="blue", params=None)

    policy_map = {
        "red_spy": lambda obs: red_spy.get_clue(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
        "blue_guess": lambda obs: blue_guess.get_guess(obs),
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="red_guess",
        policy_map=policy_map
    )

    obs = wrapped.reset()

    # Should return observation for red_guess
    assert "words" in obs
    assert "revealed" in obs
    assert obs["colors"] is None  # Guessers don't see colors
    assert "current_clue" in obs


def test_step_single_agent():
    """Test step with single agent action."""
    env = WordBatchEnv(batch_size=1, seed=42)

    red_spy = RandomSpymaster(team="red", params=None)
    blue_spy = RandomSpymaster(team="blue", params=None)
    red_guess = RandomGuesser(team="red", params=None)
    blue_guess = RandomGuesser(team="blue", params=None)

    # Focus on red_spy (spymaster)
    policy_map = {
        "red_guess": lambda obs: red_guess.get_guess(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
        "blue_guess": lambda obs: blue_guess.get_guess(obs),
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="red_spy",
        policy_map=policy_map
    )

    obs = wrapped.reset()

    # Take a step with red_spy's action
    action = {"clue": ["FIRE"], "clue_number": torch.tensor([2])}
    obs, reward, done, info = wrapped.step(action)

    # Check return types
    assert isinstance(obs, dict)
    assert isinstance(reward, torch.Tensor)
    assert isinstance(done, torch.Tensor)
    assert isinstance(info, dict)

    # Check shapes
    assert reward.shape == (1,)
    assert done.shape == (1,)


def test_other_agents_auto_controlled():
    """Test that other agents are automatically controlled."""
    env = WordBatchEnv(batch_size=1, seed=42)

    # Track which agents were called
    calls = {"red_spy": 0, "blue_spy": 0, "blue_guess": 0}

    def make_policy(agent_id, base_agent):
        def policy(obs):
            calls[agent_id] += 1
            return base_agent.get_clue(obs) if "spy" in agent_id else base_agent.get_guess(obs)
        return policy

    red_spy = RandomSpymaster(team="red", params=None)
    blue_spy = RandomSpymaster(team="blue", params=None)
    blue_guess = RandomGuesser(team="blue", params=None)

    policy_map = {
        "red_spy": make_policy("red_spy", red_spy),
        "blue_spy": make_policy("blue_spy", blue_spy),
        "blue_guess": make_policy("blue_guess", blue_guess),
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="red_guess",
        policy_map=policy_map
    )

    obs = wrapped.reset()

    # Step multiple times
    for _ in range(3):
        action = {"tile_index": torch.tensor([0])}
        obs, reward, done, info = wrapped.step(action)
        if done[0]:
            break

    # At least some of the other agents should have been called
    assert sum(calls.values()) > 0


def test_batch_size_greater_than_one():
    """Test wrapper with batch_size > 1."""
    batch_size = 4
    env = WordBatchEnv(batch_size=batch_size, seed=42)

    red_spy = RandomSpymaster(team="red", params=None)
    blue_spy = RandomSpymaster(team="blue", params=None)
    red_guess = RandomGuesser(team="red", params=None)

    policy_map = {
        "red_spy": lambda obs: red_spy.get_clue(obs),
        "red_guess": lambda obs: red_guess.get_guess(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="blue_guess",
        policy_map=policy_map
    )

    obs = wrapped.reset()

    # Observation should have batch dimension
    assert obs["revealed"].shape == (batch_size, 25)

    # Step with batched action
    action = {"tile_index": torch.zeros(batch_size, dtype=torch.int32)}
    obs, reward, done, info = wrapped.step(action)

    # Returns should have batch dimension
    assert reward.shape == (batch_size,)
    assert done.shape == (batch_size,)


def test_with_vector_env():
    """Test wrapper works with VectorBatchEnv."""
    env = VectorBatchEnv(batch_size=2, seed=42)

    # Define simple policies
    policy_map = {
        "red_spy": lambda obs: {
            "clue_vec": torch.randn(2, 384),
            "clue_number": torch.tensor([2, 2])
        },
        "red_guess": lambda obs: {
            "tile_index": torch.zeros(2, dtype=torch.int32)
        },
        "blue_spy": lambda obs: {
            "clue_vec": torch.randn(2, 384),
            "clue_number": torch.tensor([2, 2])
        },
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="blue_guess",
        policy_map=policy_map
    )

    obs = wrapped.reset()

    # Should work with vector observations
    assert "board_vectors" in obs
    assert obs["board_vectors"].shape == (2, 25, 384)

    action = {"tile_index": torch.tensor([0, 1])}
    obs, reward, done, info = wrapped.step(action)

    assert isinstance(obs, dict)
    assert reward.shape == (2,)


def test_game_state_access():
    """Test that game_state is accessible through wrapper."""
    env = WordBatchEnv(batch_size=1, seed=42)

    red_spy = RandomSpymaster(team="red", params=None)
    blue_spy = RandomSpymaster(team="blue", params=None)
    blue_guess = RandomGuesser(team="blue", params=None)

    policy_map = {
        "red_spy": lambda obs: red_spy.get_clue(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
        "blue_guess": lambda obs: blue_guess.get_guess(obs),
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="red_guess",
        policy_map=policy_map
    )

    wrapped.reset()

    # Should be able to access game state
    assert hasattr(wrapped, 'game_state')
    assert wrapped.game_state is env.game_state


def test_agent_ids_property():
    """Test agent_ids property."""
    env = WordBatchEnv(batch_size=1, seed=42)

    policy_map = {
        "red_spy": lambda obs: {},
        "red_guess": lambda obs: {},
        "blue_spy": lambda obs: {},
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="blue_guess",
        policy_map=policy_map
    )

    assert wrapped.agent_ids == env.agent_ids
    assert "blue_guess" in wrapped.agent_ids


def test_full_episode():
    """Test running a full episode through the wrapper."""
    env = WordBatchEnv(batch_size=1, seed=42)

    red_spy = RandomSpymaster(team="red", params=None)
    blue_spy = RandomSpymaster(team="blue", params=None)
    blue_guess = RandomGuesser(team="blue", params=None)

    policy_map = {
        "red_spy": lambda obs: red_spy.get_clue(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
        "blue_guess": lambda obs: blue_guess.get_guess(obs),
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="red_guess",
        policy_map=policy_map
    )

    obs = wrapped.reset()
    total_reward = 0
    steps = 0
    max_steps = 100

    while steps < max_steps:
        # Simple random action for red_guess
        action = {"tile_index": torch.randint(0, 25, (1,))}
        obs, reward, done, info = wrapped.step(action)

        total_reward += reward[0]
        steps += 1

        if done[0]:
            break

    # Game should eventually end
    assert done[0] or steps == max_steps

    # Should have accumulated some reward (positive or negative)
    assert isinstance(total_reward, (int, float, torch.Tensor))


def test_wrapper_with_embedding_agents():
    """Test wrapper with embedding agents."""
    try:
        from agents.spymaster import EmbeddingSpymaster
        from agents.guesser import EmbeddingGuesser
    except ImportError:
        pytest.skip("sentence-transformers not available")

    env = WordBatchEnv(batch_size=1, seed=42)

    # Use embedding agents for other players
    from agents.spymaster import SpymasterParams
    from agents.guesser import GuesserParams

    red_spy = EmbeddingSpymaster(team="red", params=SpymasterParams(n_candidate_clues=5, seed=42))
    blue_spy = EmbeddingSpymaster(team="blue", params=SpymasterParams(n_candidate_clues=5, seed=43))
    blue_guess = EmbeddingGuesser(team="blue", params=GuesserParams(similarity_threshold=0.0, seed=44))

    policy_map = {
        "red_spy": lambda obs: red_spy.get_clue(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
        "blue_guess": lambda obs: blue_guess.get_guess(obs),
    }

    wrapped = SingleAgentWrapper(
        env=env,
        agent_id="red_guess",
        policy_map=policy_map
    )

    obs = wrapped.reset()

    # Take a few steps
    for _ in range(3):
        action = {"tile_index": torch.tensor([0])}
        obs, reward, done, info = wrapped.step(action)
        if done[0]:
            break

    # Should work without errors
    assert isinstance(obs, dict)
