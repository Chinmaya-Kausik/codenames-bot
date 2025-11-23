"""
Tests for WordBatchEnv - word-based multi-agent environment.
"""

import torch
import pytest

from envs.word_batch_env import WordBatchEnv
from core.game_state import GameState
from core.reality_layer import create_reality_layer_from_random


def test_word_batch_env_initialization():
    """Test basic initialization."""
    batch_size = 4
    board_size = 25
    env = WordBatchEnv(batch_size=batch_size, board_size=board_size, seed=42)

    assert env.batch_size == batch_size
    assert env.board_size == board_size
    assert len(env.word_views) == batch_size
    assert len(env.current_clue_words) == batch_size
    assert env.agent_ids == ["red_spy", "red_guess", "blue_spy", "blue_guess"]


def test_reset():
    """Test environment reset."""
    env = WordBatchEnv(batch_size=2, seed=42)
    obs_dict = env.reset()

    # Check that all agents get observations
    assert set(obs_dict.keys()) == set(env.agent_ids)

    # Check observation structure for spymaster
    red_spy_obs = obs_dict["red_spy"]
    assert "words" in red_spy_obs
    assert "revealed" in red_spy_obs
    assert "colors" in red_spy_obs  # Spymasters see colors
    assert "role_encoding" in red_spy_obs

    # Check observation structure for guesser
    red_guess_obs = obs_dict["red_guess"]
    assert "words" in red_guess_obs
    assert "revealed" in red_guess_obs
    assert red_guess_obs["colors"] is None  # Guessers don't see colors
    assert "current_clue" in red_guess_obs
    assert "current_clue_number" in red_guess_obs
    assert "remaining_guesses" in red_guess_obs


def test_spymaster_observations_see_colors():
    """Test that spymasters can see colors."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    # Spymasters should see colors
    assert obs_dict["red_spy"]["colors"] is not None
    assert obs_dict["blue_spy"]["colors"] is not None
    assert obs_dict["red_spy"]["colors"].shape == (1, 25)


def test_guesser_observations_no_colors():
    """Test that guessers cannot see colors."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    # Guessers should not see colors
    assert obs_dict["red_guess"]["colors"] is None
    assert obs_dict["blue_guess"]["colors"] is None

    # But they should see clue info
    assert "current_clue" in obs_dict["red_guess"]
    assert "current_clue_number" in obs_dict["red_guess"]
    assert "remaining_guesses" in obs_dict["red_guess"]


def test_role_encoding():
    """Test role encoding in observations."""
    env = WordBatchEnv(batch_size=1, seed=42)
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


def test_spymaster_step_word_clues():
    """Test spymaster giving word-based clues."""
    batch_size = 2
    env = WordBatchEnv(batch_size=batch_size, seed=42)
    env.reset()
    device = env.device

    # Get active masks to determine which spymaster is active in each game
    active_masks = env.game_state.get_active_agent_masks()

    # Create spymaster actions for both teams
    red_clues = ["FIRE", "WATER"]
    red_clue_numbers = torch.tensor([2, 3], device=device)
    blue_clues = ["EARTH", "AIR"]
    blue_clue_numbers = torch.tensor([4, 5], device=device)

    actions_dict = {
        "red_spy": {"clue": red_clues, "clue_number": red_clue_numbers},
        "red_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_spy": {"clue": blue_clues, "clue_number": blue_clue_numbers},
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

    # Clue words should be stored
    expected_clues = [
        red_clues[i] if active_masks["red_spy"][i] else blue_clues[i]
        for i in range(batch_size)
    ]
    assert env.current_clue_words == expected_clues


def test_guesser_step_word_guesses():
    """Test guesser making word-based guesses."""
    batch_size = 1
    env = WordBatchEnv(batch_size=batch_size, seed=42)
    env.reset()

    # Determine which team is active
    active_masks = env.game_state.get_active_agent_masks()
    is_red_active = active_masks["red_spy"][0]
    active_team = "red" if is_red_active else "blue"

    # Give clue first
    actions_dict = {
        "red_spy": {"clue": ["FIRE"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.tensor([0])},
        "blue_spy": {"clue": ["WATER"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Find a team tile to guess (as a word)
    team_idx = env.game_state.current_team[0]
    team_tile_idx = torch.where(env.game_state.colors[0] == team_idx)[0][0]
    team_word = env.word_views[0].get_word(team_tile_idx)

    # Make guess using word
    actions_dict = {
        "red_spy": {"clue": [""], "clue_number": torch.zeros(batch_size, dtype=torch.int32)},
        "red_guess": {"word": [team_word]},
        "blue_spy": {"clue": [""], "clue_number": torch.zeros(batch_size, dtype=torch.int32)},
        "blue_guess": {"word": [team_word]},
    }

    obs2, rewards2, dones2, infos2 = env.step(actions_dict)

    # Tile should be revealed
    assert env.game_state.revealed[0, team_tile_idx]

    # Reward should be positive for the active team's agents (correct guess)
    active_spy = f"{active_team}_spy"
    active_guess = f"{active_team}_guess"
    assert rewards2[active_spy][0] > 0 or rewards2[active_guess][0] > 0


def test_guesser_step_index_guesses():
    """Test guesser making index-based guesses."""
    batch_size = 1
    env = WordBatchEnv(batch_size=batch_size, seed=42)
    env.reset()

    # Determine which team is active
    active_masks = env.game_state.get_active_agent_masks()
    is_red_active = active_masks["red_spy"][0]
    active_team = "red" if is_red_active else "blue"

    # Give clue first
    actions_dict = {
        "red_spy": {"clue": ["FIRE"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.tensor([0])},
        "blue_spy": {"clue": ["WATER"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Find a team tile to guess
    team_idx = env.game_state.current_team[0]
    team_tile = torch.where(env.game_state.colors[0] == team_idx)[0][0]

    # Make guess using index
    actions_dict = {
        "red_spy": {"clue": [""], "clue_number": torch.zeros(batch_size, dtype=torch.int32)},
        "red_guess": {"tile_index": torch.tensor([team_tile])},
        "blue_spy": {"clue": [""], "clue_number": torch.zeros(batch_size, dtype=torch.int32)},
        "blue_guess": {"tile_index": torch.tensor([team_tile])},
    }

    obs2, rewards2, dones2, infos2 = env.step(actions_dict)

    # Tile should be revealed
    assert env.game_state.revealed[0, team_tile]

    # Reward should be positive for the active team's agents
    active_spy = f"{active_team}_spy"
    active_guess = f"{active_team}_guess"
    assert rewards2[active_spy][0] > 0 or rewards2[active_guess][0] > 0


def test_only_active_agent_acts():
    """Test that only active agent's actions are applied."""
    batch_size = 2
    env = WordBatchEnv(batch_size=batch_size, seed=42)
    env.reset()

    # Check which spymasters are active
    active_masks = env.game_state.get_active_agent_masks()

    # Both teams give clues, but only active should be applied
    actions_dict = {
        "red_spy": {"clue": ["FIRE", "WATER"], "clue_number": torch.tensor([2, 3])},
        "red_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32)},
        "blue_spy": {"clue": ["EARTH", "AIR"], "clue_number": torch.tensor([4, 5])},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Verify only active spymaster's clue was used
    for b in range(batch_size):
        if active_masks["red_spy"][b]:
            assert env.game_state.current_clue_number[b] == 2 if b == 0 else 3
        else:
            assert env.game_state.current_clue_number[b] == 4 if b == 0 else 5


def test_rewards_only_for_active_agents():
    """Test that both roles on the same team receive the same rewards."""
    batch_size = 1
    env = WordBatchEnv(batch_size=batch_size, seed=42)
    env.reset()

    # Give clue
    active_masks = env.game_state.get_active_agent_masks()
    actions_dict = {
        "red_spy": {"clue": ["FIRE"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.tensor([0])},
        "blue_spy": {"clue": ["WATER"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # All rewards should be zero (no guess made yet)
    for agent_id in env.agent_ids:
        assert rewards[agent_id][0] == 0

    # Make a guess
    team_tile = torch.where(env.game_state.colors[0] == env.game_state.current_team[0])[0][0]
    actions_dict = {
        "red_spy": {"clue": [""], "clue_number": torch.zeros(batch_size, dtype=torch.int32)},
        "red_guess": {"tile_index": torch.tensor([team_tile])},
        "blue_spy": {"clue": [""], "clue_number": torch.zeros(batch_size, dtype=torch.int32)},
        "blue_guess": {"tile_index": torch.tensor([team_tile])},
    }

    obs2, rewards2, dones2, infos2 = env.step(actions_dict)

    # Determine which guesser was active
    active_guesser_masks = env.game_state.get_active_agent_masks()

    # Both roles on the same team should receive the same reward
    if active_guesser_masks["red_guess"][0]:
        # Red was active, both red agents get same reward
        assert rewards2["red_spy"][0] == rewards2["red_guess"][0]
        assert rewards2["red_spy"][0] > 0  # Positive for revealing team tile
        # Blue team gets negative reward (opponent revealed a tile)
        assert rewards2["blue_spy"][0] == rewards2["blue_guess"][0]
        assert rewards2["blue_spy"][0] < 0
    else:
        # Blue was active, both blue agents get same reward
        assert rewards2["blue_spy"][0] == rewards2["blue_guess"][0]
        assert rewards2["blue_spy"][0] > 0
        # Red team gets negative reward
        assert rewards2["red_spy"][0] == rewards2["red_guess"][0]
        assert rewards2["red_spy"][0] < 0


def test_with_reality_layer():
    """Test environment with reality layer for token lookup."""
    from core.clue_vocab import ClueVocab
    from core.reality_layer import RealityLayer

    batch_size = 1

    # Create reality layer with tokens
    vocab_size = 100
    embedding_dim = 384
    tokens = [f"WORD_{i}" for i in range(vocab_size)]

    # Create random vectors
    torch.manual_seed(42)
    vectors = torch.randn(vocab_size, embedding_dim)
    vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)  # Normalize

    # Create ClueVocab with tokens
    clue_vocab = ClueVocab(vectors=vectors, tokens=tokens)
    reality_layer = RealityLayer(clue_vocab, enabled=True)

    env = WordBatchEnv(batch_size=batch_size, reality_layer=reality_layer, seed=42)
    env.reset()
    device = env.device

    # Give word clue from the vocabulary
    clue_word = tokens[0]  # Use a word from the vocab
    actions_dict = {
        "red_spy": {"clue": [clue_word], "clue_number": torch.tensor([2], device=device)},
        "red_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
        "blue_spy": {"clue": [clue_word], "clue_number": torch.tensor([2], device=device)},
        "blue_guess": {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=device)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Clue should be stored as word
    assert len(env.current_clue_words[0]) > 0

    # Info should include clue words
    assert "clue_words" in infos["red_spy"]

    # The clue word should match what we gave
    assert env.current_clue_words[0] == clue_word


def test_dones_when_game_over():
    """Test that done flags are set when game ends."""
    env = WordBatchEnv(batch_size=1, seed=42)
    env.reset()

    # Manually set game to be over
    env.game_state.game_over[0] = True
    env.game_state.winner[0] = GameState.RED

    # Get dones
    dones = env._get_dones()

    # All agents should see game as done
    for agent_id in env.agent_ids:
        assert dones[agent_id][0] == True


def test_infos_dict():
    """Test info dictionary structure."""
    env = WordBatchEnv(batch_size=2, seed=42)
    obs = env.reset()

    # Give clues
    actions_dict = {
        "red_spy": {"clue": ["FIRE", "WATER"], "clue_number": torch.tensor([2, 3])},
        "red_guess": {"tile_index": torch.zeros(2, dtype=torch.int32)},
        "blue_spy": {"clue": ["EARTH", "AIR"], "clue_number": torch.tensor([4, 5])},
        "blue_guess": {"tile_index": torch.zeros(2, dtype=torch.int32)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Check info structure
    for agent_id in env.agent_ids:
        info = infos[agent_id]
        assert "winner" in info
        assert "turn_count" in info
        assert "unrevealed_counts" in info
        assert "clue_words" in info
        assert "revealed_words" in info

        # Check types
        assert isinstance(info["winner"], torch.Tensor)
        assert isinstance(info["turn_count"], torch.Tensor)
        assert isinstance(info["clue_words"], list)
        assert isinstance(info["revealed_words"], list)


def test_words_in_observations():
    """Test that words are included in observations."""
    batch_size = 2
    env = WordBatchEnv(batch_size=batch_size, seed=42)
    obs_dict = env.reset()

    for agent_id in env.agent_ids:
        obs = obs_dict[agent_id]
        assert "words" in obs
        assert isinstance(obs["words"], list)
        assert len(obs["words"]) == batch_size

        # Each game should have board_size words
        for game_words in obs["words"]:
            assert len(game_words) == env.board_size
            assert all(isinstance(w, str) for w in game_words)


def test_clue_in_guesser_observations():
    """Test that guessers see the current clue."""
    env = WordBatchEnv(batch_size=1, seed=42)
    env.reset()

    # Give a clue
    actions_dict = {
        "red_spy": {"clue": ["FIRE"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.tensor([0])},
        "blue_spy": {"clue": ["WATER"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
    }

    obs, rewards, dones, infos = env.step(actions_dict)

    # Guessers should see the clue
    assert "current_clue" in obs["red_guess"]
    assert "current_clue" in obs["blue_guess"]

    # The active team's guesser should see a non-empty clue
    active_masks = env.game_state.get_active_agent_masks()
    if active_masks["red_spy"][0]:
        # Red gave clue, now it's red guesser's turn (after phase transition)
        # But we need to check the observation from the guesser's perspective
        pass

    # At least one clue should be set
    assert len(env.current_clue_words[0]) > 0


def test_custom_word_pool():
    """Test using a custom word pool."""
    custom_pool = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"] * 10  # 50 words
    env = WordBatchEnv(batch_size=1, word_pool=custom_pool, seed=42)
    obs = env.reset()

    # All words should be from custom pool
    game_words = obs["red_spy"]["words"][0]
    for word in game_words:
        assert word in custom_pool


def test_batch_independence():
    """Test that games in batch are independent."""
    batch_size = 3
    env = WordBatchEnv(batch_size=batch_size, seed=42)
    env.reset()

    # Each game should have different words
    all_words = [env.word_views[b].get_all_words() for b in range(batch_size)]

    # At least some words should be different between games
    # (extremely unlikely for all games to have identical word sets)
    assert all_words[0] != all_words[1] or all_words[1] != all_words[2]


def test_invalid_word_guess_raises_error():
    """Test that guessing an invalid word raises a clear error."""
    env = WordBatchEnv(batch_size=1, seed=42)
    env.reset()

    # Give a spymaster clue first
    actions_dict = {
        "red_spy": {"clue": ["TEST"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.tensor([0])},
        "blue_spy": {"clue": ["WORD"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.tensor([0])},
    }
    env.step(actions_dict)

    # Now try to guess with an invalid word
    actions_dict = {
        "red_spy": {"clue": ["TEST"], "clue_number": torch.tensor([2])},
        "red_guess": {"word": ["NOTAWORD"]},  # Invalid word not on board
        "blue_spy": {"clue": ["WORD"], "clue_number": torch.tensor([2])},
        "blue_guess": {"word": ["ALSONOTAWORD"]},
    }

    with pytest.raises(ValueError, match="Invalid guess word"):
        env.step(actions_dict)


def test_lowercase_word_guess_works():
    """Test that lowercase word guesses work (case-insensitive)."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs = env.reset()

    # Get a valid word from the board
    valid_word = obs["red_spy"]["words"][0][0]  # First word on board

    # Give a spymaster clue first
    actions_dict = {
        "red_spy": {"clue": ["TEST"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.tensor([0])},
        "blue_spy": {"clue": ["WORD"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.tensor([0])},
    }
    env.step(actions_dict)

    # Now guess with lowercase version
    actions_dict = {
        "red_spy": {"clue": ["TEST"], "clue_number": torch.tensor([2])},
        "red_guess": {"word": [valid_word.lower()]},  # Lowercase should work
        "blue_spy": {"clue": ["WORD"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.tensor([1])},
    }

    # Should not raise
    obs, rewards, dones, infos = env.step(actions_dict)


def test_observation_mutation_doesnt_affect_env():
    """Test that mutating observation lists doesn't corrupt env state."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs = env.reset()

    # Get words from observation
    words_before = env.word_views[0].get_all_words()

    # Try to mutate the observation (should fail since it's a tuple)
    with pytest.raises((TypeError, AttributeError)):
        obs["red_spy"]["words"][0][0] = "HACKED"

    # Verify words are unchanged
    words_after = env.word_views[0].get_all_words()
    assert words_before == words_after


def test_finished_game_preserves_clues():
    """Test that finished games preserve clues instead of zeroing them."""
    env = WordBatchEnv(batch_size=2, seed=42)
    env.reset()

    # Force finish one game by setting game_over
    env.game_state.game_over[0] = True
    env.game_state.winner[0] = GameState.RED

    # Set a clue for the finished game
    env.current_clue_words = ["TESTCLUE", ""]
    env.game_state.current_clue_number[0] = 3

    # Now try to give clues (only game 1 should update)
    actions_dict = {
        "red_spy": {"clue": ["NEWCLUE", "NEWCLUE"], "clue_number": torch.tensor([2, 2])},
        "red_guess": {"tile_index": torch.tensor([0, 0])},
        "blue_spy": {"clue": ["BLUECLUE", "BLUECLUE"], "clue_number": torch.tensor([2, 2])},
        "blue_guess": {"tile_index": torch.tensor([0, 0])},
    }

    obs, _, _, _ = env.step(actions_dict)

    # Game 0 should preserve its clue (game is finished)
    # This test will depend on which team is active in game 1
    assert env.current_clue_words[0] == "TESTCLUE"  # Preserved
    assert env.game_state.current_clue_number[0] == 3  # Preserved
