"""
Experiments module for Codenames.

This module provides experiment orchestration tools for running multi-agent
Codenames games, parameter sweeps, and collecting experimental results with
flexible tracking.

The experiment framework supports:
- Multi-agent environments (WordBatchEnv, VectorBatchEnv)
- Callback-based tracking (Summary, Episode, Trajectory trackers)
- Parameter sweeps for optimization
- Batched parallel game execution

Exported Classes:
    MultiAgentCodenamesExperiment: Main experiment orchestration class
    GameTracker: Abstract base class for trackers
    SummaryTracker: Aggregate statistics tracker (O(1) memory)
    EpisodeTracker: Per-episode results tracker (O(n_games) memory)
    TrajectoryTracker: Full trajectory tracker (O(n_games × n_steps) memory)

Example:
    >>> from experiments import MultiAgentCodenamesExperiment, SummaryTracker
    >>> from envs import WordBatchEnv
    >>> from agents import RandomSpymaster, RandomGuesser
    >>>
    >>> # Create experiment
    >>> exp = MultiAgentCodenamesExperiment(
    ...     env_factory=lambda seed: WordBatchEnv(batch_size=8, seed=seed),
    ...     max_turns=50
    ... )
    >>>
    >>> # Define policy map
    >>> policy_map = {
    ...     "red_spy": lambda obs: RandomSpymaster(team="red").get_clue(obs),
    ...     "red_guess": lambda obs: RandomGuesser(team="red").get_guess(obs),
    ...     "blue_spy": lambda obs: RandomSpymaster(team="blue").get_clue(obs),
    ...     "blue_guess": lambda obs: RandomGuesser(team="blue").get_guess(obs),
    ... }
    >>>
    >>> # Run with tracking
    >>> tracker = SummaryTracker()
    >>> results = exp.run_games(
    ...     policy_map=policy_map,
    ...     n_games=100,
    ...     tracker=tracker,
    ...     seed=42
    ... )
    >>>
    >>> # Analyze results
    >>> print(f"Red win rate: {results['red_win_rate']:.2%}")
"""

from experiments.trackers import GameTracker, SummaryTracker, EpisodeTracker, TrajectoryTracker
from experiments.multi_agent_experiment import MultiAgentCodenamesExperiment

__all__ = [
    "MultiAgentCodenamesExperiment",
    "GameTracker",
    "SummaryTracker",
    "EpisodeTracker",
    "TrajectoryTracker",
]
