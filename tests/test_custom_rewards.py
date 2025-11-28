"""
Tests for custom reward functions in both WordBatchEnv and VectorBatchEnv.
"""

import pytest
import torch
import numpy as np

from envs.word_batch_env import WordBatchEnv, default_sparse_reward, default_dense_reward
from envs.vector_batch_env import VectorBatchEnv
from core.game_state import GameState
from agents.spymaster import RandomSpymaster
from agents.guesser import RandomGuesser


def dense_reward_fn(prev_state: GameState, new_state: GameState, agent_id: str, team_idx: int) -> torch.Tensor:
    """
    Dense reward function for testing.

    Rewards:
    - +1 for each own team tile revealed
    - -1 for each opponent tile revealed
    - -10 for assassin
    - +10 for winning
    - -10 for losing
    """
    batch_size = new_state.game_over.shape[0]
    device = new_state.game_over.device
    rewards = torch.zeros(batch_size, device=device)

    # Get team/opponent labels
    if team_idx == GameState.RED:
        team_label = 0
        opp_label = 1
    else:
        team_label = 1
        opp_label = 0

    # Tiles revealed this step
    revealed_diff = new_state.revealed & ~prev_state.revealed

    # Team tiles revealed (+1 each)
    team_tiles = (new_state.colors == team_label) & revealed_diff
    rewards += team_tiles.sum(dim=1).float()

    # Opponent tiles revealed (-1 each)
    opp_tiles = (new_state.colors == opp_label) & revealed_diff
    rewards -= opp_tiles.sum(dim=1).float()

    # Assassin revealed (-10)
    assassin_tiles = (new_state.colors == 3) & revealed_diff
    rewards -= 10.0 * assassin_tiles.sum(dim=1).float()

    # Game end rewards
    newly_finished = ~prev_state.game_over & new_state.game_over
    won_games = newly_finished & (new_state.winner == team_idx)
    lost_games = newly_finished & (new_state.winner != team_idx) & (new_state.winner >= 0)

    rewards += 10.0 * won_games.float()
    rewards -= 10.0 * lost_games.float()

    return rewards


def constant_reward_fn(prev_state: GameState, new_state: GameState, agent_id: str, team_idx: int) -> torch.Tensor:
    """Constant reward of 1.0 for testing."""
    batch_size = new_state.game_over.shape[0]
    return torch.ones(batch_size, device=new_state.game_over.device)


class TestDefaultSparseReward:
    """Test the default sparse reward function."""

    def test_default_sparse_reward_no_game_end(self):
        """Test default reward when game continues."""
        batch_size = 4
        device = "cpu"

        # Create two states - no game over
        prev_state = GameState(batch_size=batch_size, board_size=25, device=device)
        prev_state.reset()

        new_state = GameState(batch_size=batch_size, board_size=25, device=device)
        new_state.colors = prev_state.colors.clone()
        new_state.revealed = prev_state.revealed.clone()
        new_state.revealed[0, 0] = True  # Reveal one tile
        new_state.game_over = torch.zeros(batch_size, dtype=torch.bool)

        # Get rewards
        rewards = default_sparse_reward(prev_state, new_state, "red_spy", GameState.RED)

        # Should all be zero (no game ended)
        assert rewards.shape == (batch_size,)
        assert torch.all(rewards == 0.0)

    def test_default_sparse_reward_win(self):
        """Test default reward when team wins."""
        batch_size = 4
        device = "cpu"

        # Create states where red wins in game 0
        prev_state = GameState(batch_size=batch_size, board_size=25, device=device)
        prev_state.reset()
        prev_state.game_over = torch.zeros(batch_size, dtype=torch.bool)

        new_state = GameState(batch_size=batch_size, board_size=25, device=device)
        new_state.colors = prev_state.colors.clone()
        new_state.revealed = prev_state.revealed.clone()
        new_state.game_over = torch.zeros(batch_size, dtype=torch.bool)
        new_state.game_over[0] = True
        new_state.winner = torch.full((batch_size,), -1, dtype=torch.int32)
        new_state.winner[0] = GameState.RED

        # Red team should get +1 for game 0
        rewards = default_sparse_reward(prev_state, new_state, "red_spy", GameState.RED)
        assert rewards[0] == 1.0
        assert torch.all(rewards[1:] == 0.0)

        # Blue team should get -1 for game 0
        rewards = default_sparse_reward(prev_state, new_state, "blue_spy", GameState.BLUE)
        assert rewards[0] == -1.0
        assert torch.all(rewards[1:] == 0.0)

    def test_default_sparse_reward_batched(self):
        """Test default reward with multiple games ending."""
        batch_size = 4
        device = "cpu"

        prev_state = GameState(batch_size=batch_size, board_size=25, device=device)
        prev_state.reset()
        prev_state.game_over = torch.zeros(batch_size, dtype=torch.bool)

        new_state = GameState(batch_size=batch_size, board_size=25, device=device)
        new_state.colors = prev_state.colors.clone()
        new_state.revealed = prev_state.revealed.clone()
        # Games 0, 1 end - game 0: red wins, game 1: blue wins
        new_state.game_over = torch.tensor([True, True, False, False])
        new_state.winner = torch.tensor([0, 1, -1, -1], dtype=torch.int32)  # 0=red, 1=blue

        # Red team: +1 for game 0, -1 for game 1, 0 for others
        rewards = default_sparse_reward(prev_state, new_state, "red_spy", GameState.RED)
        assert rewards[0] == 1.0
        assert rewards[1] == -1.0
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0


class TestWordBatchEnvCustomRewards:
    """Test custom reward functions in WordBatchEnv."""

    def test_default_reward_used(self):
        """Test that default reward is used when none provided."""
        env = WordBatchEnv(batch_size=2, seed=42)
        assert env.reward_fn == default_dense_reward

    def test_custom_reward_applied(self):
        """Test that custom reward function is actually called."""
        env = WordBatchEnv(batch_size=2, seed=42, reward_fn=constant_reward_fn)

        obs = env.reset()

        # Create actions for all agents
        actions = {
            "red_spy": {"clue": ["TEST", "TEST"], "clue_number": torch.tensor([1, 1])},
            "red_guess": {"tile_index": torch.tensor([0, 0])},
            "blue_spy": {"clue": ["BLUE", "BLUE"], "clue_number": torch.tensor([1, 1])},
            "blue_guess": {"tile_index": torch.tensor([1, 1])},
        }

        # Step (spymaster phase)
        obs, rewards, dones, infos = env.step(actions)

        # Rewards should all be 0 in spymaster phase
        assert all(torch.all(rewards[aid] == 0) for aid in env.agent_ids)

        # Step again (guesser phase)
        obs, rewards, dones, infos = env.step(actions)

        # Now constant_reward_fn should be called, returning 1.0 for all
        for aid in env.agent_ids:
            assert torch.all(rewards[aid] == 1.0), f"Expected constant reward of 1.0 for {aid}"

    def test_dense_reward_function(self):
        """Test dense reward function that rewards per-tile progress."""
        env = WordBatchEnv(batch_size=2, seed=42, reward_fn=dense_reward_fn)
        obs = env.reset()

        # Force board to have known colors for predictable testing
        # Game 0: tile 0 is red (team 0)
        env.game_state.colors[0, 0] = 0  # Red
        # Game 1: tile 0 is blue (team 1)
        env.game_state.colors[1, 0] = 1  # Blue

        actions_spy = {
            "red_spy": {"clue": ["TEST", "TEST"], "clue_number": torch.tensor([1, 1])},
            "red_guess": {"tile_index": torch.tensor([0, 0])},
            "blue_spy": {"clue": ["BLUE", "BLUE"], "clue_number": torch.tensor([1, 1])},
            "blue_guess": {"tile_index": torch.tensor([1, 1])},
        }

        # Spy phase
        obs, rewards, dones, infos = env.step(actions_spy)

        # Guesser phase - red team guesses tile 0 in both games
        obs, rewards, dones, infos = env.step(actions_spy)

        # Red team in game 0: should get +1 (revealed own tile)
        # Red team in game 1: should get -1 (revealed opponent tile)
        # (Unless game ended, then +10 or -10 override)

        # Just check that rewards are not all zeros (dense reward is being used)
        assert not torch.all(rewards["red_spy"] == 0.0), "Dense rewards should be non-zero"

    def test_reward_signature(self):
        """Test that reward function receives correct arguments."""
        called_args = {}

        def tracking_reward_fn(prev_state, new_state, agent_id, team_idx):
            called_args["prev_state"] = prev_state
            called_args["new_state"] = new_state
            called_args["agent_id"] = agent_id
            called_args["team_idx"] = team_idx
            batch_size = new_state.game_over.shape[0]
            return torch.zeros(batch_size, device=new_state.game_over.device)

        env = WordBatchEnv(batch_size=2, seed=42, reward_fn=tracking_reward_fn)
        env.reset()

        actions = {
            "red_spy": {"clue": ["TEST", "TEST"], "clue_number": torch.tensor([1, 1])},
            "red_guess": {"tile_index": torch.tensor([0, 0])},
            "blue_spy": {"clue": ["BLUE", "BLUE"], "clue_number": torch.tensor([1, 1])},
            "blue_guess": {"tile_index": torch.tensor([1, 1])},
        }

        env.step(actions)  # Spy phase
        env.step(actions)  # Guesser phase - this calls reward function

        # Check that reward function was called with correct types
        assert isinstance(called_args["prev_state"], GameState)
        assert isinstance(called_args["new_state"], GameState)
        assert called_args["agent_id"] in env.agent_ids
        assert called_args["team_idx"] in [0, 1]


class TestVectorBatchEnvCustomRewards:
    """Test custom reward functions in VectorBatchEnv."""

    def test_default_reward_used(self):
        """Test that default reward is used when none provided."""
        env = VectorBatchEnv(batch_size=2, seed=42)
        assert env.reward_fn == default_dense_reward

    def test_custom_reward_applied(self):
        """Test that custom reward function is actually called."""
        env = VectorBatchEnv(batch_size=2, seed=42, reward_fn=constant_reward_fn)

        obs = env.reset()

        # Create actions
        actions = {
            "red_spy": {
                "clue_vec": torch.randn(2, env.embedding_dim),
                "clue_number": torch.tensor([1, 1])
            },
            "red_guess": {"tile_index": torch.tensor([0, 0])},
            "blue_spy": {
                "clue_vec": torch.randn(2, env.embedding_dim),
                "clue_number": torch.tensor([1, 1])
            },
            "blue_guess": {"tile_index": torch.tensor([1, 1])},
        }

        # Step (spymaster phase)
        obs, rewards, dones, infos = env.step(actions)

        # Rewards should all be 0 in spymaster phase
        assert all(torch.all(rewards[aid] == 0) for aid in env.agent_ids)

        # Step again (guesser phase)
        obs, rewards, dones, infos = env.step(actions)

        # Now constant_reward_fn should be called
        for aid in env.agent_ids:
            assert torch.all(rewards[aid] == 1.0), f"Expected constant reward of 1.0 for {aid}"

    def test_dense_reward_function(self):
        """Test dense reward function works with VectorBatchEnv."""
        env = VectorBatchEnv(batch_size=2, seed=42, reward_fn=dense_reward_fn)
        env.reset()

        # Set known colors
        env.game_state.colors[0, 0] = 0  # Red
        env.game_state.colors[1, 0] = 1  # Blue

        actions = {
            "red_spy": {
                "clue_vec": torch.randn(2, env.embedding_dim),
                "clue_number": torch.tensor([1, 1])
            },
            "red_guess": {"tile_index": torch.tensor([0, 0])},
            "blue_spy": {
                "clue_vec": torch.randn(2, env.embedding_dim),
                "clue_number": torch.tensor([1, 1])
            },
            "blue_guess": {"tile_index": torch.tensor([1, 1])},
        }

        env.step(actions)  # Spy phase
        obs, rewards, dones, infos = env.step(actions)  # Guesser phase

        # Dense rewards should be non-zero
        assert not torch.all(rewards["red_spy"] == 0.0), "Dense rewards should be non-zero"


class TestRewardFunctionIntegration:
    """Integration tests for reward functions across full games."""

    def test_full_game_with_custom_rewards(self):
        """Test a full game with custom rewards to ensure no errors."""
        env = WordBatchEnv(batch_size=2, seed=42, reward_fn=dense_reward_fn)

        agents = {
            "red_spy": RandomSpymaster(team="red"),
            "red_guess": RandomGuesser(team="red"),
            "blue_spy": RandomSpymaster(team="blue"),
            "blue_guess": RandomGuesser(team="blue"),
        }

        obs = env.reset(seed=42)

        for turn in range(50):
            if torch.all(env.game_state.game_over):
                break

            actions = {
                aid: agent.get_clue(obs[aid]) if "spy" in aid else agent.get_guess(obs[aid])
                for aid, agent in agents.items()
            }

            obs, rewards, dones, infos = env.step(actions)

            # Rewards should be valid tensors
            for aid in env.agent_ids:
                assert rewards[aid].shape == (2,)
                assert not torch.isnan(rewards[aid]).any()
                assert not torch.isinf(rewards[aid]).any()

        # At least one game should have ended
        assert torch.any(env.game_state.game_over)

    def test_reward_accumulation(self):
        """Test that rewards accumulate correctly over multiple steps."""
        env = WordBatchEnv(batch_size=1, seed=42, reward_fn=dense_reward_fn)

        agents = {
            "red_spy": RandomSpymaster(team="red"),
            "red_guess": RandomGuesser(team="red"),
            "blue_spy": RandomSpymaster(team="blue"),
            "blue_guess": RandomGuesser(team="blue"),
        }

        obs = env.reset(seed=42)
        total_rewards = {aid: 0.0 for aid in env.agent_ids}

        for turn in range(20):
            if env.game_state.game_over[0]:
                break

            actions = {
                aid: agent.get_clue(obs[aid]) if "spy" in aid else agent.get_guess(obs[aid])
                for aid, agent in agents.items()
            }

            obs, rewards, dones, infos = env.step(actions)

            for aid in env.agent_ids:
                total_rewards[aid] += rewards[aid][0].item()

        # Both team members should get same reward
        assert total_rewards["red_spy"] == total_rewards["red_guess"], "Same team members should get same reward"
        assert total_rewards["blue_spy"] == total_rewards["blue_guess"], "Same team members should get same reward"

        # Winner should have positive total reward, loser negative (with dense rewards)
        if env.game_state.game_over[0]:
            red_total = total_rewards["red_spy"]
            blue_total = total_rewards["blue_spy"]

            if env.game_state.winner[0] == 0:  # Red won
                assert red_total > blue_total, "Winner should have higher total reward"
            elif env.game_state.winner[0] == 1:  # Blue won
                assert blue_total > red_total, "Winner should have higher total reward"
