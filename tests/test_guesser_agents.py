"""
Tests for guesser agents.
"""

import torch
import pytest

from agents.guesser import BaseGuesser, GuesserParams, RandomGuesser
from envs.word_batch_env import WordBatchEnv

# Try to import EmbeddingGuesser
try:
    from agents.guesser import EmbeddingGuesser
    from agents.spymaster import EmbeddingSpymaster
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    EmbeddingGuesser = None
    EmbeddingSpymaster = None

skip_if_no_embeddings = pytest.mark.skipif(
    not EMBEDDING_AVAILABLE,
    reason="sentence-transformers not available"
)


def test_random_guesser_initialization():
    """Test RandomGuesser initialization."""
    guesser = RandomGuesser(team="red")
    assert guesser.team == "red"
    assert guesser.params is not None
    assert isinstance(guesser.params, GuesserParams)


def test_random_guesser_get_guess():
    """Test RandomGuesser get_guess method."""
    env = WordBatchEnv(batch_size=2, seed=42)
    obs_dict = env.reset()

    # Give clue first
    actions_dict = {
        "red_spy": {"clue": ["FIRE", "WATER"], "clue_number": torch.tensor([2, 2])},
        "red_guess": {"tile_index": torch.zeros(2, dtype=torch.int32)},
        "blue_spy": {"clue": ["EARTH", "AIR"], "clue_number": torch.tensor([2, 2])},
        "blue_guess": {"tile_index": torch.zeros(2, dtype=torch.int32)},
    }
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Now make a guess
    guesser = RandomGuesser(team="red", params=GuesserParams(seed=42))
    action = guesser.get_guess(obs_dict["red_guess"])

    # Check action structure
    assert "tile_index" in action
    assert isinstance(action["tile_index"], torch.Tensor)
    assert action["tile_index"].shape == (2,)


def test_random_guesser_with_env():
    """Test RandomGuesser works with environment."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    # Give clue
    actions_dict = {
        "red_spy": {"clue": ["FIRE"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
        "blue_spy": {"clue": ["WATER"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
    }
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Create guesser
    guesser = RandomGuesser(team="blue", params=GuesserParams(seed=42))

    # Get action
    action = guesser.get_guess(obs_dict["blue_guess"])

    # Build actions dict
    actions_dict = {
        "red_spy": {"clue": [""], "clue_number": torch.zeros(1, dtype=torch.int32)},
        "red_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
        "blue_spy": {"clue": [""], "clue_number": torch.zeros(1, dtype=torch.int32)},
        "blue_guess": action,
    }

    # Step environment
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Check that a tile was revealed
    assert torch.any(env.game_state.revealed)


@skip_if_no_embeddings
def test_embedding_guesser_initialization():
    """Test EmbeddingGuesser initialization."""
    guesser = EmbeddingGuesser(team="red")
    assert guesser.team == "red"
    assert guesser.params is not None
    assert guesser.model is not None


@skip_if_no_embeddings
def test_embedding_guesser_get_guess():
    """Test EmbeddingGuesser get_guess method."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    # Give clue first
    actions_dict = {
        "red_spy": {"clue": ["FIRE"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
        "blue_spy": {"clue": ["WATER"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
    }
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Create guesser
    guesser = EmbeddingGuesser(team="red", params=GuesserParams(similarity_threshold=0.0))
    action = guesser.get_guess(obs_dict["red_guess"])

    # Check action structure
    assert "tile_index" in action
    assert isinstance(action["tile_index"], torch.Tensor)
    assert action["tile_index"].shape == (1,)


@skip_if_no_embeddings
def test_embedding_guesser_with_env():
    """Test EmbeddingGuesser works with environment."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    # Give clue
    actions_dict = {
        "red_spy": {"clue": ["ANIMAL"], "clue_number": torch.tensor([2])},
        "red_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
        "blue_spy": {"clue": ["WATER"], "clue_number": torch.tensor([2])},
        "blue_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
    }
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Create guesser with low threshold
    guesser = EmbeddingGuesser(team="blue", params=GuesserParams(similarity_threshold=0.0))

    # Get action
    action = guesser.get_guess(obs_dict["blue_guess"])

    # Build actions dict
    actions_dict = {
        "red_spy": {"clue": [""], "clue_number": torch.zeros(1, dtype=torch.int32)},
        "red_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
        "blue_spy": {"clue": [""], "clue_number": torch.zeros(1, dtype=torch.int32)},
        "blue_guess": action,
    }

    # Step environment
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Check that a tile was revealed
    assert torch.any(env.game_state.revealed)


def test_guesser_params():
    """Test GuesserParams dataclass."""
    params = GuesserParams(
        similarity_threshold=0.4,
        confidence_threshold=0.6,
        seed=42
    )

    assert params.similarity_threshold == 0.4
    assert params.confidence_threshold == 0.6
    assert params.seed == 42


@skip_if_no_embeddings
def test_embedding_guesser_batch():
    """Test EmbeddingGuesser with batch > 1."""
    env = WordBatchEnv(batch_size=3, seed=42)
    obs_dict = env.reset()

    # Give clues
    actions_dict = {
        "red_spy": {"clue": ["FIRE", "WATER", "EARTH"], "clue_number": torch.tensor([2, 2, 2])},
        "red_guess": {"tile_index": torch.zeros(3, dtype=torch.int32)},
        "blue_spy": {"clue": ["AIR", "METAL", "WOOD"], "clue_number": torch.tensor([2, 2, 2])},
        "blue_guess": {"tile_index": torch.zeros(3, dtype=torch.int32)},
    }
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Create guesser
    guesser = EmbeddingGuesser(team="red", params=GuesserParams(similarity_threshold=0.0))
    action = guesser.get_guess(obs_dict["red_guess"])

    # Check action structure for batch
    assert action["tile_index"].shape == (3,)


@skip_if_no_embeddings
def test_full_game_with_agents():
    """Test a full game turn with embedding agents."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    # Create agents
    from agents.spymaster import EmbeddingSpymaster, SpymasterParams
    red_spy = EmbeddingSpymaster(team="red", params=SpymasterParams(n_candidate_clues=5, seed=42))
    blue_spy = EmbeddingSpymaster(team="blue", params=SpymasterParams(n_candidate_clues=5, seed=43))
    red_guesser = EmbeddingGuesser(team="red", params=GuesserParams(similarity_threshold=0.0, seed=44))
    blue_guesser = EmbeddingGuesser(team="blue", params=GuesserParams(similarity_threshold=0.0, seed=45))

    # Play one turn
    # Step 1: Spymasters give clues
    actions_dict = {
        "red_spy": red_spy.get_clue(obs_dict["red_spy"]),
        "red_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
        "blue_spy": blue_spy.get_clue(obs_dict["blue_spy"]),
        "blue_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
    }
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Step 2: Guessers make guesses
    actions_dict = {
        "red_spy": {"clue": [""], "clue_number": torch.zeros(1, dtype=torch.int32)},
        "red_guess": red_guesser.get_guess(obs_dict["red_guess"]),
        "blue_spy": {"clue": [""], "clue_number": torch.zeros(1, dtype=torch.int32)},
        "blue_guess": blue_guesser.get_guess(obs_dict["blue_guess"]),
    }
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Check that at least one tile was revealed
    assert torch.any(env.game_state.revealed)
