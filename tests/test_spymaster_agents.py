"""
Tests for spymaster agents.
"""

import torch
import pytest

from agents.spymaster import BaseSpymaster, SpymasterParams, RandomSpymaster
from envs.word_batch_env import WordBatchEnv

# Try to import EmbeddingSpymaster
try:
    from agents.spymaster import EmbeddingSpymaster
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    EmbeddingSpymaster = None

skip_if_no_embeddings = pytest.mark.skipif(
    not EMBEDDING_AVAILABLE,
    reason="sentence-transformers not available"
)


def test_random_spymaster_initialization():
    """Test RandomSpymaster initialization."""
    spy = RandomSpymaster(team="red")
    assert spy.team == "red"
    assert spy.params is not None
    assert isinstance(spy.params, SpymasterParams)


def test_random_spymaster_get_clue():
    """Test RandomSpymaster get_clue method."""
    env = WordBatchEnv(batch_size=2, seed=42)
    obs_dict = env.reset()

    spy = RandomSpymaster(team="red", params=SpymasterParams(seed=42))
    action = spy.get_clue(obs_dict["red_spy"])

    # Check action structure
    assert "clue" in action
    assert "clue_number" in action
    assert isinstance(action["clue"], list)
    assert isinstance(action["clue_number"], torch.Tensor)
    assert len(action["clue"]) == 2  # batch_size
    assert action["clue_number"].shape == (2,)


def test_random_spymaster_with_env():
    """Test RandomSpymaster works with environment."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    # Create agents
    red_spy = RandomSpymaster(team="red", params=SpymasterParams(seed=42))
    blue_spy = RandomSpymaster(team="blue", params=SpymasterParams(seed=43))

    # Get actions from both spymasters
    red_action = red_spy.get_clue(obs_dict["red_spy"])
    blue_action = blue_spy.get_clue(obs_dict["blue_spy"])

    # Build actions dict
    actions_dict = {
        "red_spy": red_action,
        "red_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
        "blue_spy": blue_action,
        "blue_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
    }

    # Step environment
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Check that we transitioned to guesser phase
    from core.game_state import GameState
    assert env.game_state.phase[0] == GameState.GUESSER_PHASE


@skip_if_no_embeddings
def test_embedding_spymaster_initialization():
    """Test EmbeddingSpymaster initialization."""
    spy = EmbeddingSpymaster(team="red")
    assert spy.team == "red"
    assert spy.params is not None
    assert spy.model is not None
    assert spy.clue_word_pool is not None
    assert len(spy.clue_word_pool) > 0


@skip_if_no_embeddings
def test_embedding_spymaster_get_clue():
    """Test EmbeddingSpymaster get_clue method."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    spy = EmbeddingSpymaster(team="red", params=SpymasterParams(seed=42, n_candidate_clues=10))
    action = spy.get_clue(obs_dict["red_spy"])

    # Check action structure
    assert "clue" in action
    assert "clue_number" in action
    assert isinstance(action["clue"], list)
    assert isinstance(action["clue_number"], torch.Tensor)
    assert len(action["clue"]) == 1  # batch_size
    assert action["clue_number"].shape == (1,)
    assert action["clue_number"][0] >= 1
    assert action["clue_number"][0] <= 3


@skip_if_no_embeddings
def test_embedding_spymaster_with_env():
    """Test EmbeddingSpymaster works with environment."""
    env = WordBatchEnv(batch_size=1, seed=42)
    obs_dict = env.reset()

    # Create spymaster
    spy = EmbeddingSpymaster(team="blue", params=SpymasterParams(seed=42, n_candidate_clues=10))

    # Get action
    action = spy.get_clue(obs_dict["blue_spy"])

    # Build actions dict
    actions_dict = {
        "red_spy": {"clue": ["DUMMY"], "clue_number": torch.tensor([1], dtype=torch.int32)},
        "red_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
        "blue_spy": action,
        "blue_guess": {"tile_index": torch.zeros(1, dtype=torch.int32)},
    }

    # Step environment
    obs_dict, rewards, dones, infos = env.step(actions_dict)

    # Check that clue was stored
    assert "clue_words" in infos["blue_spy"]


@skip_if_no_embeddings
def test_custom_clue_pool():
    """Test EmbeddingSpymaster with custom clue pool."""
    custom_pool = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"] * 20
    params = SpymasterParams(clue_word_pool=custom_pool, n_candidate_clues=5)

    spy = EmbeddingSpymaster(team="red", params=params)
    assert spy.clue_word_pool == custom_pool


def test_spymaster_params():
    """Test SpymasterParams dataclass."""
    params = SpymasterParams(
        n_candidate_clues=100,
        risk_tolerance=3.0,
        opponent_penalty=1.5,
        seed=42
    )

    assert params.n_candidate_clues == 100
    assert params.risk_tolerance == 3.0
    assert params.opponent_penalty == 1.5
    assert params.seed == 42


@skip_if_no_embeddings
def test_embedding_spymaster_batch():
    """Test EmbeddingSpymaster with batch > 1."""
    env = WordBatchEnv(batch_size=3, seed=42)
    obs_dict = env.reset()

    spy = EmbeddingSpymaster(team="red", params=SpymasterParams(seed=42, n_candidate_clues=5))
    action = spy.get_clue(obs_dict["red_spy"])

    # Check action structure for batch
    assert len(action["clue"]) == 3
    assert action["clue_number"].shape == (3,)
    assert all(1 <= n <= 3 for n in action["clue_number"])
