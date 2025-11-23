"""
Tests for envs.vector_batch_env module.
"""

import torch
import pytest

from envs.vector_batch_env import VectorBatchEnv
from core.reality_layer import create_reality_layer_from_random
from core.game_state import GameState
from utils.device import get_device


def test_vector_batch_env_initialization():
    """Test basic initialization."""
    batch_size = 4
    env = VectorBatchEnv(batch_size=batch_size, seed=42)

    assert env.batch_size == batch_size
    assert env.board_size == 25
    assert env.embedding_dim == 384
    assert env.board_vectors.shape == (batch_size, 25, 384)
    assert len(env.agent_ids) == 4
    assert set(env.agent_ids) == {"red_spy", "red_guess", "blue_spy", "blue_guess"}


def test_reset():
    """Test reset functionality."""
    batch_size = 2
    env = VectorBatchEnv(batch_size=batch_size, seed=42)

    obs_dict = env.reset(seed=99)

    # Check all agent IDs present
    assert set(obs_dict.keys()) == set(env.agent_ids)

    # Check observation shapes
    for agent_id, obs in obs_dict.items():
        assert obs["board_vectors"].shape == (batch_size, 25, 384)
        assert obs["revealed"].shape == (batch_size, 25)
        assert obs["role_encoding"].shape == (batch_size, 4)

    # Check no tiles revealed initially
    assert not torch.any(obs_dict["red_spy"]["revealed"])


def test_spymaster_observations_see_colors():
    """Test that spymasters can see colors."""
    batch_size = 2
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    obs_dict = env.reset()

    # Spymasters should see colors
    assert obs_dict["red_spy"]["colors"] is not None
    assert obs_dict["blue_spy"]["colors"] is not None

    assert obs_dict["red_spy"]["colors"].shape == (batch_size, 25)
    assert obs_dict["blue_spy"]["colors"].shape == (batch_size, 25)


def test_guesser_observations_no_colors():
    """Test that guessers cannot see colors."""
    batch_size = 2
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    obs_dict = env.reset()

    # Guessers should NOT see colors
    assert obs_dict["red_guess"]["colors"] is None
    assert obs_dict["blue_guess"]["colors"] is None

    # But should see clue info
    assert "current_clue_vec" in obs_dict["red_guess"]
    assert "current_clue_number" in obs_dict["red_guess"]
    assert "remaining_guesses" in obs_dict["red_guess"]


def test_role_encoding():
    """Test role encoding in observations."""
    batch_size = 1
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    obs_dict = env.reset()
    device = env.device

    # Red spymaster: [1, 0, 1, 0]
    assert torch.equal(obs_dict["red_spy"]["role_encoding"][0], torch.tensor([1, 0, 1, 0], device=device))

    # Red guesser: [1, 0, 0, 1]
    assert torch.equal(obs_dict["red_guess"]["role_encoding"][0], torch.tensor([1, 0, 0, 1], device=device))

    # Blue spymaster: [0, 1, 1, 0]
    assert torch.equal(obs_dict["blue_spy"]["role_encoding"][0], torch.tensor([0, 1, 1, 0], device=device))

    # Blue guesser: [0, 1, 0, 1]
    assert torch.equal(obs_dict["blue_guess"]["role_encoding"][0], torch.tensor([0, 1, 0, 1], device=device))


def test_spymaster_step():
    """Test spymaster giving clue."""
    batch_size = 2
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    env.reset()
    device = env.device

    # Get active masks to determine which spymaster is active in each game
    active_masks = env.game_state.get_active_agent_masks()

    # Create spymaster actions for both teams
    red_clue_vecs = torch.randn(batch_size, env.embedding_dim, device=device)
    red_clue_numbers = torch.tensor([2, 3], device=device)
    blue_clue_vecs = torch.randn(batch_size, env.embedding_dim, device=device)
    blue_clue_numbers = torch.tensor([4, 5], device=device)

    actions_dict = {
        "red_spy": {"clue_vec": red_clue_vecs, "clue_number": red_clue_numbers},
        "red_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_spy": {"clue_vec": blue_clue_vecs, "clue_number": blue_clue_numbers},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Phase should transition to guesser
    assert torch.all(env.game_state.phase == GameState.GUESSER_PHASE)

    # Clue numbers should be set according to which spymaster was active
    expected_clue_numbers = torch.where(active_masks["red_spy"], red_clue_numbers, blue_clue_numbers)
    assert torch.equal(env.game_state.current_clue_number, expected_clue_numbers)

    # Remaining guesses should be set
    assert torch.equal(env.game_state.remaining_guesses, expected_clue_numbers + 1)


def test_guesser_step():
    """Test guesser making a guess."""
    batch_size = 1
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    env.reset()
    device = env.device

    # Determine which team is active
    active_masks = env.game_state.get_active_agent_masks()
    is_red_active = active_masks["red_spy"][0]
    active_team = "red" if is_red_active else "blue"

    # Give clue first
    clue_vec = torch.randn(batch_size, env.embedding_dim, device=device)
    actions_dict = {
        "red_spy": {"clue_vec": clue_vec, "clue_number": torch.tensor([2], device=device)},
        "red_guess": {"tile_index": torch.tensor([0], device=device)},
        "blue_spy": {"clue_vec": clue_vec, "clue_number": torch.tensor([2], device=device)},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Find a team tile to guess
    team_tile = torch.where(env.game_state.colors[0] == env.game_state.current_team[0])[0][0]

    # Make guess
    actions_dict = {
        "red_spy": {"clue_vec": torch.zeros((batch_size, env.embedding_dim), device=device), "clue_number": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "red_guess": {"tile_index": torch.tensor([team_tile], device=device)},
        "blue_spy": {"clue_vec": torch.zeros((batch_size, env.embedding_dim), device=device), "clue_number": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_guess": {"tile_index": torch.tensor([team_tile], device=device)},
    }

    obs2, rewards2, dones2, infos2 = env.step(actions_dict)

    # Tile should be revealed
    assert env.game_state.revealed[0, team_tile]

    # Reward should be positive for the active team's agents (correct guess)
    active_spy = f"{active_team}_spy"
    active_guess = f"{active_team}_guess"
    assert rewards2[active_spy][0] > 0 or rewards2[active_guess][0] > 0


def test_only_active_agent_acts():
    """Test that only active agent's actions are applied."""
    batch_size = 2
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    env.reset()
    device = env.device

    # Set up so game 0 is red's turn, game 1 is blue's turn
    env.game_state.current_team[0] = GameState.RED
    env.game_state.current_team[1] = GameState.BLUE

    clue_vec = torch.randn(batch_size, env.embedding_dim, device=device)

    # Both spymasters give clues
    actions_dict = {
        "red_spy": {"clue_vec": clue_vec, "clue_number": torch.tensor([2, 2], device=device)},
        "blue_spy": {"clue_vec": clue_vec, "clue_number": torch.tensor([3, 3], device=device)},
        "red_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Game 0 should use red's clue (number=2)
    # Game 1 should use blue's clue (number=3)
    assert env.game_state.current_clue_number[0] == 2
    assert env.game_state.current_clue_number[1] == 3


def test_rewards_only_for_active_agents():
    """Test that only active agents get non-zero rewards."""
    batch_size = 2
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    env.reset()
    device = env.device

    # Give clue
    clue_vec = torch.randn(batch_size, env.embedding_dim, device=device)
    actions_dict = {
        "red_spy": {"clue_vec": clue_vec, "clue_number": torch.tensor([2, 2], device=device)},
        "red_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_spy": {"clue_vec": torch.zeros((batch_size, env.embedding_dim), device=device), "clue_number": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Spymaster rewards should be zero (not active during guess phase)
    assert torch.all(rewards["red_spy"] == 0)
    assert torch.all(rewards["blue_spy"] == 0)


def test_with_reality_layer():
    """Test environment with reality layer enabled."""
    batch_size = 2
    reality_layer = create_reality_layer_from_random(
        vocab_size=100,
        embedding_dim=384,
        enabled=True,
        seed=42
    )

    env = VectorBatchEnv(
        batch_size=batch_size,
        embedding_dim=384,
        reality_layer=reality_layer,
        seed=42
    )

    env.reset()
    device = env.device

    # Give clue
    clue_vec = torch.randn(batch_size, 384, device=device)
    actions_dict = {
        "red_spy": {"clue_vec": clue_vec, "clue_number": torch.tensor([2, 2], device=device)},
        "red_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_spy": {"clue_vec": torch.zeros((batch_size, 384), device=device), "clue_number": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Clue should be snapped to vocab
    assert torch.all(env.game_state.current_clue_index >= 0)
    assert torch.all(env.game_state.current_clue_index < 100)


def test_dones_when_game_over():
    """Test done flags when games end."""
    batch_size = 1
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    env.reset()

    # Manually set game over
    env.game_state.game_over[0] = True
    env.game_state.winner[0] = GameState.RED

    obs = env._get_observations()
    dones = env._get_dones()

    # All agents should see done=True
    for agent_id in env.agent_ids:
        assert dones[agent_id][0]


def test_infos_dict():
    """Test that infos dict contains expected information."""
    batch_size = 2
    env = VectorBatchEnv(batch_size=batch_size, seed=42)
    env.reset()
    device = env.device

    clue_vec = torch.randn(batch_size, env.embedding_dim, device=device)
    actions_dict = {
        "red_spy": {"clue_vec": clue_vec, "clue_number": torch.tensor([2, 2], device=device)},
        "red_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_spy": {"clue_vec": torch.zeros((batch_size, env.embedding_dim), device=device), "clue_number": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Check infos structure
    for agent_id in env.agent_ids:
        assert "winner" in infos[agent_id]
        assert "turn_count" in infos[agent_id]
        assert "unrevealed_counts" in infos[agent_id]

        # Check shapes
        assert infos[agent_id]["winner"].shape == (batch_size,)
        assert infos[agent_id]["turn_count"].shape == (batch_size,)


def test_malformed_clue_vec_raises_error():
    """Test that malformed clue vectors raise descriptive errors."""
    env = VectorBatchEnv(batch_size=2, embedding_dim=384, seed=42)
    env.reset()

    # Try to give a clue with wrong shape
    actions_dict = {
        "red_spy": {
            "clue_vec": torch.randn(2, 100),  # Wrong embedding_dim
            "clue_number": torch.tensor([2, 2])
        },
        "red_guess": {"tile_index": torch.tensor([0, 0])},
        "blue_spy": {
            "clue_vec": torch.randn(2, 384),
            "clue_number": torch.tensor([2, 2])
        },
        "blue_guess": {"tile_index": torch.tensor([0, 0])},
    }

    with pytest.raises(ValueError, match="provided clue_vec with shape"):
        env.step(actions_dict)


def test_finished_game_preserves_clue_vectors():
    """Test that finished games preserve clue vectors instead of zeroing them."""
    env = VectorBatchEnv(batch_size=2, embedding_dim=384, seed=42)
    env.reset()

    # Force finish one game
    from core.game_state import GameState
    env.game_state.game_over[0] = True
    env.game_state.winner[0] = GameState.RED

    # Set a clue vector for the finished game
    test_vec = torch.randn(384, device=env.device)
    env.current_clue_vectors[0] = test_vec
    env.game_state.current_clue_number[0] = 3

    # Now try to give clues (only game 1 should update)
    actions_dict = {
        "red_spy": {
            "clue_vec": torch.randn(2, 384, device=env.device),
            "clue_number": torch.tensor([2, 2], device=env.device)
        },
        "red_guess": {"tile_index": torch.tensor([0, 0], device=env.device)},
        "blue_spy": {
            "clue_vec": torch.randn(2, 384, device=env.device),
            "clue_number": torch.tensor([2, 2], device=env.device)
        },
        "blue_guess": {"tile_index": torch.tensor([0, 0], device=env.device)},
    }

    env.step(actions_dict)

    # Game 0 should preserve its clue vector
    assert torch.allclose(env.current_clue_vectors[0], test_vec)
    assert env.game_state.current_clue_number[0] == 3


def test_observation_mutation_doesnt_affect_env():
    """Test that mutating observation tensors doesn't corrupt env state."""
    env = VectorBatchEnv(batch_size=1, seed=42)
    env.reset()

    # Get observations
    obs = env._get_observations()

    # Get the original board vector value from env
    original_value = env.board_vectors[0, 0].clone()

    # Mutate the observation (should not affect env since it's a clone)
    obs["red_spy"]["board_vectors"][0, 0] = torch.zeros_like(obs["red_spy"]["board_vectors"][0, 0])

    # Verify env board_vectors are unchanged
    assert torch.allclose(env.board_vectors[0, 0], original_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
