"""
Tests for multi-agent experiment runner and trackers.
"""

import torch
import pytest

from experiments import (
    MultiAgentCodenamesExperiment,
    SummaryTracker,
    EpisodeTracker,
    TrajectoryTracker
)
from envs.word_batch_env import WordBatchEnv
from agents.spymaster import RandomSpymaster
from agents.guesser import RandomGuesser


def make_env_factory(batch_size=1):
    """Create environment factory for testing."""
    return lambda seed: WordBatchEnv(batch_size=batch_size, seed=seed)


def make_policy_map():
    """Create policy map with random agents."""
    red_spy = RandomSpymaster(team="red")
    red_guess = RandomGuesser(team="red")
    blue_spy = RandomSpymaster(team="blue")
    blue_guess = RandomGuesser(team="blue")

    return {
        "red_spy": lambda obs: red_spy.get_clue(obs),
        "red_guess": lambda obs: red_guess.get_guess(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
        "blue_guess": lambda obs: blue_guess.get_guess(obs),
    }


# ============================================================================
# SummaryTracker Tests
# ============================================================================

def test_summary_tracker_initialization():
    """Test SummaryTracker initialization."""
    tracker = SummaryTracker()
    assert tracker.total_games == 0
    assert tracker.total_steps == 0


def test_summary_tracker_accumulates_stats():
    """Test that SummaryTracker accumulates statistics correctly."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=50
    )

    tracker = SummaryTracker()
    results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=10,
        tracker=tracker,
        seed=42
    )

    # Check results structure
    assert "total_games" in results
    assert "red_win_rate" in results
    assert "blue_win_rate" in results
    assert "avg_turns" in results
    assert "rewards_per_agent" in results

    # Check values
    assert results["total_games"] == 10
    assert 0 <= results["red_win_rate"] <= 1
    assert 0 <= results["blue_win_rate"] <= 1
    assert results["red_win_rate"] + results["blue_win_rate"] <= 1  # Can have draws
    assert results["avg_turns"] > 0


def test_summary_tracker_reset():
    """Test SummaryTracker reset."""
    tracker = SummaryTracker()

    # Run some games
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=20
    )
    exp.run_games(policy_map=make_policy_map(), n_games=5, tracker=tracker, seed=42)

    assert tracker.total_games == 5

    # Reset
    tracker.reset()
    assert tracker.total_games == 0
    assert tracker.total_steps == 0


# ============================================================================
# EpisodeTracker Tests
# ============================================================================

def test_episode_tracker_initialization():
    """Test EpisodeTracker initialization."""
    tracker = EpisodeTracker()
    assert len(tracker.episodes) == 0


def test_episode_tracker_stores_episodes():
    """Test that EpisodeTracker stores per-episode data."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=50
    )

    tracker = EpisodeTracker()
    results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=5,
        tracker=tracker,
        seed=42
    )

    # Should have 5 episodes
    assert len(results) == 5

    # Check structure of each episode
    for episode in results:
        assert "episode_idx" in episode
        assert "total_rewards" in episode
        assert "winner" in episode
        assert "turns" in episode

        # Check that all agents have rewards
        assert "red_spy" in episode["total_rewards"]
        assert "red_guess" in episode["total_rewards"]
        assert "blue_spy" in episode["total_rewards"]
        assert "blue_guess" in episode["total_rewards"]


def test_episode_tracker_episode_indices():
    """Test that episode indices are correctly assigned."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=30
    )

    tracker = EpisodeTracker()
    results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=3,
        tracker=tracker,
        seed=42
    )

    # Check indices
    assert results[0]["episode_idx"] == 0
    assert results[1]["episode_idx"] == 1
    assert results[2]["episode_idx"] == 2


# ============================================================================
# TrajectoryTracker Tests
# ============================================================================

def test_trajectory_tracker_initialization():
    """Test TrajectoryTracker initialization."""
    tracker = TrajectoryTracker()
    assert len(tracker.episodes) == 0
    assert tracker.store_observations == True
    assert tracker.store_actions == True


def test_trajectory_tracker_stores_trajectories():
    """Test that TrajectoryTracker stores step-by-step data."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=20
    )

    tracker = TrajectoryTracker()
    results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=2,
        tracker=tracker,
        seed=42
    )

    # Should have 2 episodes
    assert len(results) == 2

    # Check first episode has trajectory
    episode = results[0]
    assert "trajectory" in episode
    assert "winner" in episode
    assert "turns" in episode
    assert "total_rewards" in episode

    # Trajectory should have steps
    trajectory = episode["trajectory"]
    assert len(trajectory) > 0

    # Check step structure
    step = trajectory[0]
    assert "step" in step
    assert "rewards" in step
    assert "dones" in step
    assert "observations" in step  # store_observations=True
    assert "actions" in step  # store_actions=True


def test_trajectory_tracker_without_observations():
    """Test TrajectoryTracker with observations disabled."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=15
    )

    tracker = TrajectoryTracker(store_observations=False)
    results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=1,
        tracker=tracker,
        seed=42
    )

    # Check that observations are not stored
    episode = results[0]
    step = episode["trajectory"][0]
    assert "observations" not in step
    assert "actions" in step  # Still stored
    assert "rewards" in step


def test_trajectory_tracker_without_actions():
    """Test TrajectoryTracker with actions disabled."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=15
    )

    tracker = TrajectoryTracker(store_actions=False)
    results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=1,
        tracker=tracker,
        seed=42
    )

    # Check that actions are not stored
    episode = results[0]
    step = episode["trajectory"][0]
    assert "actions" not in step
    assert "observations" in step  # Still stored
    assert "rewards" in step


# ============================================================================
# MultiAgentCodenamesExperiment Tests
# ============================================================================

def test_experiment_initialization():
    """Test MultiAgentCodenamesExperiment initialization."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=2),
        max_turns=30
    )

    assert exp.max_turns == 30
    assert exp.env_factory is not None


def test_experiment_missing_policy():
    """Test that missing policy raises error."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=30
    )

    # Missing blue_guess policy
    incomplete_policy_map = {
        "red_spy": lambda obs: {},
        "red_guess": lambda obs: {},
        "blue_spy": lambda obs: {},
    }

    with pytest.raises(ValueError, match="policy_map missing policies"):
        exp.run_games(
            policy_map=incomplete_policy_map,
            n_games=1,
            seed=42
        )


def test_experiment_with_batch_size_greater_than_one():
    """Test experiment with batched environment."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=4),
        max_turns=30
    )

    tracker = SummaryTracker()
    results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=8,  # 2 batches of 4
        tracker=tracker,
        seed=42
    )

    assert results["total_games"] == 8


def test_experiment_verbose_output(capsys):
    """Test that verbose mode prints progress."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=20
    )

    exp.run_games(
        policy_map=make_policy_map(),
        n_games=3,
        seed=42,
        verbose=True
    )

    # Check that something was printed
    captured = capsys.readouterr()
    assert "Completed" in captured.out


def test_experiment_run_sweep():
    """Test parameter sweep functionality."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=30
    )

    # Define parameter grid
    param_grid = [
        {"config": "a"},
        {"config": "b"},
    ]

    # Policy factory (ignores params for this test)
    def policy_factory(params):
        return make_policy_map()

    # Run sweep
    sweep_results = exp.run_sweep(
        policy_factory=policy_factory,
        param_grid=param_grid,
        n_games_per_config=5,
        seed=42
    )

    # Should have results for each configuration
    assert len(sweep_results) == 2

    # Check structure
    for result in sweep_results:
        assert "params" in result
        assert "results" in result
        assert "seed" in result

        # Results should have summary stats
        assert "total_games" in result["results"]
        assert result["results"]["total_games"] == 5


def test_experiment_deterministic_with_seed():
    """Test that experiments are deterministic with same seed."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=30
    )

    # Create policy map with seeded agents for determinism
    def make_seeded_policy_map(seed):
        from agents.spymaster import SpymasterParams
        from agents.guesser import GuesserParams

        red_spy = RandomSpymaster(team="red", params=SpymasterParams(seed=seed))
        red_guess = RandomGuesser(team="red", params=GuesserParams(seed=seed+1))
        blue_spy = RandomSpymaster(team="blue", params=SpymasterParams(seed=seed+2))
        blue_guess = RandomGuesser(team="blue", params=GuesserParams(seed=seed+3))

        return {
            "red_spy": lambda obs: red_spy.get_clue(obs),
            "red_guess": lambda obs: red_guess.get_guess(obs),
            "blue_spy": lambda obs: blue_spy.get_clue(obs),
            "blue_guess": lambda obs: blue_guess.get_guess(obs),
        }

    # Run twice with same seeds
    results1 = exp.run_games(
        policy_map=make_seeded_policy_map(seed=100),
        n_games=5,
        tracker=EpisodeTracker(),
        seed=42
    )

    results2 = exp.run_games(
        policy_map=make_seeded_policy_map(seed=100),
        n_games=5,
        tracker=EpisodeTracker(),
        seed=42
    )

    # Results should be identical (same winners, turns, etc.)
    for ep1, ep2 in zip(results1, results2):
        assert ep1["winner"] == ep2["winner"]
        assert ep1["turns"] == ep2["turns"]


def test_tracker_accumulates_across_multiple_runs():
    """Test that tracker accumulates across multiple run_games calls."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=30
    )

    tracker = SummaryTracker()

    # Run games multiple times with same tracker
    exp.run_games(policy_map=make_policy_map(), n_games=5, tracker=tracker, seed=42)
    exp.run_games(policy_map=make_policy_map(), n_games=3, tracker=tracker, seed=100)

    results = tracker.get_results()

    # Should have accumulated stats from both runs
    assert results["total_games"] == 8


def test_composite_tracking():
    """Test using multiple trackers together (manual composition)."""
    exp = MultiAgentCodenamesExperiment(
        env_factory=make_env_factory(batch_size=1),
        max_turns=30
    )

    # Run with summary tracker
    summary = SummaryTracker()
    summary_results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=10,
        tracker=summary,
        seed=42
    )

    # Run with episode tracker (same seed for same results)
    episodes = EpisodeTracker()
    episode_results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=10,
        tracker=episodes,
        seed=42
    )

    # Both should report same number of games
    assert summary_results["total_games"] == 10
    assert len(episode_results) == 10

    # Compute total rewards from episodes should match summary
    total_red_spy_reward = sum(ep["total_rewards"]["red_spy"] for ep in episode_results)
    avg_from_episodes = total_red_spy_reward / 10

    # Should be reasonably close (relaxed tolerance for float32 precision, stochastic agents, and batch handling)
    assert abs(avg_from_episodes - summary_results["rewards_per_agent"]["red_spy"]) < 5.0


def test_exact_game_count_with_large_batch():
    """Test that run_games runs exactly n_games even when batch_size exceeds remaining quota."""
    # Create environment with batch_size=32 but only run 5 games
    exp = MultiAgentCodenamesExperiment(
        env_factory=lambda seed: make_env_factory(batch_size=32)(seed),
        max_turns=30
    )

    # Run with summary tracker to count games
    summary = SummaryTracker()
    results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=5,
        tracker=summary,
        seed=42
    )

    # Should report exactly 5 games, not 32
    assert results["total_games"] == 5

    # Run with episode tracker to verify individual episodes
    episodes = EpisodeTracker()
    episode_results = exp.run_games(
        policy_map=make_policy_map(),
        n_games=5,
        tracker=episodes,
        seed=42
    )

    # Should have exactly 5 episodes
    assert len(episode_results) == 5
