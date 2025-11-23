# Codenames Bot

Multi-agent Codenames environment with support for LLM and deep RL agents.

## Overview

This project provides a complete multi-agent implementation of the board game Codenames, designed for:
- **Deep reinforcement learning research**: Vectorized environments with batched execution
- **LLM agent development**: Word-based environments compatible with language models
- **Hybrid agent systems**: Reality layer for mixing continuous and discrete agents
- **Parameter optimization**: Built-in experiment framework with tracking and sweeps

## Key Features

- ✅ **Multi-agent API**: Fixed agent IDs for spymasters and guessers on both teams
- ✅ **Dual representations**: Word-based (LLMs) and vector-based (neural networks)
- ✅ **Batched execution**: Run multiple games in parallel for efficient training
- ✅ **Reality layer**: Optional vocabulary snapping for continuous→discrete conversion
- ✅ **Flexible rewards**: Customizable reward functions (default: sparse win/loss)
- ✅ **Experiment framework**: Parameter sweeps, tracking, and visualization
- ✅ **128 passing tests**: Comprehensive test coverage across all modules

## Architecture

```
codenames-bot/
├── core/                  # Shared game logic (representation-agnostic)
│   ├── game_state.py     # Batched game state management
│   ├── clue_vocab.py     # Clue vocabulary for reality layer
│   └── reality_layer.py  # Optional vocab snapping
│
├── views/                 # Representation mappings
│   ├── word_view.py      # Tile ID ↔ word mapping
│   └── vector_view.py    # Tile ID ↔ embedding mapping
│
├── envs/                  # Multi-agent environments
│   ├── word_batch_env.py # Word-based (for LLMs/humans)
│   ├── vector_batch_env.py # Vector-based (for deep RL)
│   └── wrappers/
│       └── single_agent_wrapper.py # Single-agent training wrapper
│
├── agents/                # Agent implementations
│   ├── spymaster/        # Spymaster agents (give clues)
│   │   ├── base_spymaster.py
│   │   ├── random_spymaster.py
│   │   └── embedding_spymaster.py
│   └── guesser/          # Guesser agents (select tiles)
│       ├── base_guesser.py
│       ├── random_guesser.py
│       └── embedding_guesser.py
│
├── experiments/           # Experiment orchestration
│   ├── trackers.py       # Data collection (Summary, Episode, Trajectory)
│   └── multi_agent_experiment.py # Experiment runner
│
├── tests/                 # Unit tests (128 tests)
├── notebooks/             # Interactive demos (4 notebooks)
└── utils/                 # Utilities
    └── wordlist.py       # Word pools
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/codenames-bot.git
cd codenames-bot

# Install dependencies
pip install numpy sentence-transformers pytest torch

# Run tests
python3 -m pytest tests/ -v
```

### GPU Acceleration

**The entire codebase uses PyTorch** with automatic GPU acceleration:

- ✅ **CUDA** (NVIDIA GPUs)
- ✅ **MPS** (Apple Silicon M1/M2/M3)
- ✅ **CPU** fallback

All components run on the same device for zero-copy performance:
- **Core game state** (PyTorch tensors)
- **Environments** (batched tensor operations)
- **Agents** (embedding models and inference)
- **Views** (vector/word representations)

**Performance (Apple M2 Pro, MPS):**
```
Batch Size | Games/sec  | Speedup vs B=1
-----------+------------+----------------
    1      |    100     |     1.0x
   32      |   1,200    |    12.0x
  128      |   1,892    |    18.9x
```

**Key Benefits:**
- Zero CPU↔GPU transfers during training
- Full batched tensor operations throughout
- Optimal GPU utilization at larger batch sizes

**Device Management:**
```python
# Automatic device detection
from utils.device import get_device
device = get_device()  # Returns cuda/mps/cpu

# Or specify manually
env = WordBatchEnv(batch_size=32, device="cuda")
```

**Design Rationale:**
- **Core & Environments**: PyTorch tensors for batched game logic on GPU
- **Agents**: PyTorch models for inference on same device
- **Trackers**: Convert to NumPy for CPU-side analysis (plotting, saving, statistics)

This design keeps compute-intensive operations on GPU while moving results to CPU only at the experiment interface.

### Word-based Environment (for LLMs)

```python
from envs.word_batch_env import WordBatchEnv
from agents.spymaster import RandomSpymaster
from agents.guesser import RandomGuesser

# Create environment
env = WordBatchEnv(batch_size=1, seed=42)

# Create agents
red_spy = RandomSpymaster(team="red")
red_guess = RandomGuesser(team="red")
blue_spy = RandomSpymaster(team="blue")
blue_guess = RandomGuesser(team="blue")

# Reset environment
obs_dict = env.reset()

# Game loop
for turn in range(50):
    if env.game_state.game_over[0]:
        break

    # Get actions from all agents
    actions_dict = {
        "red_spy": red_spy.get_clue(obs_dict["red_spy"]),
        "red_guess": red_guess.get_guess(obs_dict["red_guess"]),
        "blue_spy": blue_spy.get_clue(obs_dict["blue_spy"]),
        "blue_guess": blue_guess.get_guess(obs_dict["blue_guess"]),
    }

    # Step environment
    obs_dict, rewards_dict, dones_dict, infos_dict = env.step(actions_dict)

# Check winner
winner = infos_dict['red_spy']['winner'][0]
print(f"Winner: {'Red' if winner == 0 else 'Blue' if winner == 1 else 'None'}")
```

### Vector-based Environment (for Deep RL)

```python
from envs.vector_batch_env import VectorBatchEnv
from core.reality_layer import create_reality_layer_from_random

# Create reality layer (optional)
reality_layer = create_reality_layer_from_random(
    vocab_size=1000,
    embedding_dim=384,
    seed=42
)

# Create environment with batching
env = VectorBatchEnv(
    batch_size=32,  # 32 parallel games
    reality_layer=reality_layer,
    seed=42
)

# ... training loop with neural network policies
```

### Running Experiments

```python
from experiments import MultiAgentCodenamesExperiment, SummaryTracker
from envs.word_batch_env import WordBatchEnv

# Create experiment
exp = MultiAgentCodenamesExperiment(
    env_factory=lambda seed: WordBatchEnv(batch_size=8, seed=seed),
    max_turns=50
)

# Define policy map
policy_map = {
    "red_spy": lambda obs: red_spy.get_clue(obs),
    "red_guess": lambda obs: red_guess.get_guess(obs),
    "blue_spy": lambda obs: blue_spy.get_clue(obs),
    "blue_guess": lambda obs: blue_guess.get_guess(obs),
}

# Run games with tracking
tracker = SummaryTracker()
results = exp.run_games(
    policy_map=policy_map,
    n_games=100,
    tracker=tracker,
    seed=42,
    verbose=True
)

print(f"Red win rate: {results['red_win_rate']:.2%}")
print(f"Average turns: {results['avg_turns']:.1f}")
```

### Single-Agent Training

```python
from envs.wrappers import SingleAgentWrapper

# Wrap environment to focus on training red_guess
env = SingleAgentWrapper(
    env=WordBatchEnv(batch_size=4, seed=42),
    agent_id="red_guess",  # Focus on this agent
    policy_map={
        "red_spy": lambda obs: red_spy.get_clue(obs),
        "blue_spy": lambda obs: blue_spy.get_clue(obs),
        "blue_guess": lambda obs: blue_guess.get_guess(obs),
    }
)

# Standard single-agent RL interface
obs = env.reset()
obs, reward, done, info = env.step(action)  # Only red_guess's action needed
```

## Multi-Agent API

Both `WordBatchEnv` and `VectorBatchEnv` expose the same multi-agent interface:

### Agent IDs
Fixed agent IDs: `["red_spy", "red_guess", "blue_spy", "blue_guess"]`

### Reset
```python
obs_dict: dict[str, dict] = env.reset(seed=42)
# Returns observations for all four agents
```

### Step
```python
import torch

actions_dict = {
    "red_spy": {"clue": ["FIRE"], "clue_number": torch.tensor([2], dtype=torch.int32)},
    "red_guess": {"tile_index": torch.tensor([5], dtype=torch.int32)},
    "blue_spy": {"clue": ["WATER"], "clue_number": torch.tensor([3], dtype=torch.int32)},
    "blue_guess": {"tile_index": torch.tensor([12], dtype=torch.int32)},
}

obs_dict, rewards_dict, dones_dict, infos_dict = env.step(actions_dict)
```

### Turn-Based Execution
- Only ONE agent is active per game per step
- Active agents determined by game phase (spymaster vs guesser) and current team
- Inactive agents' actions are ignored, and they receive zero reward
- Check active agents with: `env.game_state.get_active_agent_masks()`

### Role-Aware Observations
Each agent receives different observations:
- **Spymasters** see board colors (know which tiles belong to which team)
- **Guessers** see current clue and remaining guesses (don't see colors)
- All agents see: revealed tiles, role encoding, current team, phase

## Custom Rewards

Both environments support custom reward functions:

```python
import torch

def custom_reward_fn(prev_state, new_state, agent_id, team_idx):
    """
    Custom reward function.

    Args:
        prev_state: GameState before action (PyTorch tensors)
        new_state: GameState after action (PyTorch tensors)
        agent_id: Agent ID (e.g., "red_spy")
        team_idx: Team index (0=red, 1=blue)

    Returns:
        Float reward
    """
    # Win reward
    if new_state.game_over and new_state.winner == team_idx:
        return 1.0

    # Dense shaping: reward for revealing team tiles
    revealed_diff = new_state.revealed - prev_state.revealed
    team_tiles_revealed = torch.sum(revealed_diff & (new_state.colors == team_idx))

    return 0.1 * team_tiles_revealed.item()

# Use in environment
env = WordBatchEnv(batch_size=32, reward_fn=custom_reward_fn)
```

**Note**: All GameState attributes are now PyTorch tensors. Use `.item()` to extract Python scalars when needed.

## Reality Layer

The reality layer bridges continuous neural network outputs and discrete vocabulary:

```python
from core.reality_layer import create_reality_layer_from_random

# Create reality layer with 1000-word vocabulary
reality_layer = create_reality_layer_from_random(
    vocab_size=1000,
    embedding_dim=384,
    seed=42
)

# Use in environment
env = VectorBatchEnv(batch_size=32, reality_layer=reality_layer)

# Neural agent outputs continuous vector
clue_vector = neural_net(obs)  # Shape: [B, 384]

# Reality layer snaps to nearest vocabulary word
# Environment automatically handles this when reality_layer is provided
```

**Benefits:**
- Preserves combinatorial hardness of word selection
- Enables hybrid LLM + neural agent play
- Can be toggled on/off for speed vs. realism tradeoff

## Experiment Framework

### Trackers

Three pre-built tracker types for data collection:

#### SummaryTracker (O(1) memory)
Computes aggregate statistics: win rates, average rewards, game length
```python
tracker = SummaryTracker()
results = exp.run_games(policy_map, n_games=100, tracker=tracker)
# Returns: {total_games, red_win_rate, blue_win_rate, avg_turns, rewards_per_agent}
```

#### EpisodeTracker (O(n_games) memory)
Stores per-episode results: total rewards, winner, turns
```python
tracker = EpisodeTracker()
episodes = exp.run_games(policy_map, n_games=50, tracker=tracker)
# Returns: list of episode dicts
```

#### TrajectoryTracker (O(n_games × n_steps) memory)
Stores full step-by-step data: observations, actions, rewards
```python
tracker = TrajectoryTracker(store_observations=True, store_actions=True)
trajectories = exp.run_games(policy_map, n_games=10, tracker=tracker)
# Returns: list of trajectory dicts with full step data
```

### Parameter Sweeps

```python
def make_policies(params):
    """Create policy map from parameters."""
    spy = EmbeddingSpymaster(
        team="red",
        params=SpymasterParams(n_candidate_clues=params['n_candidates'])
    )
    # ... create other agents
    return policy_map

# Define parameter grid
param_grid = [
    {"n_candidates": 10},
    {"n_candidates": 50},
    {"n_candidates": 100},
]

# Run sweep
results = exp.run_sweep(
    policy_factory=make_policies,
    param_grid=param_grid,
    n_games_per_config=50,
    seed=42
)

# Analyze results
for r in results:
    print(f"n_candidates={r['params']['n_candidates']}: "
          f"red_wr={r['results']['red_win_rate']:.2%}")
```

## Demo Notebooks

Explore the `notebooks/` directory for interactive examples:

1. **01_getting_started.ipynb** - Basic environment and agent usage
2. **02_multi_agent_experiments.ipynb** - Running experiments with trackers
3. **03_training_single_agent.ipynb** - Using SingleAgentWrapper for RL
4. **04_parameter_sweeps.ipynb** - Parameter optimization examples

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_word_batch_env.py -v

# Run with coverage
python3 -m pytest tests/ --cov=. --cov-report=html
```

**Test Coverage:**
- Core modules: 100% (game_state, clue_vocab, reality_layer, views)
- Environments: >90% (word_batch_env, vector_batch_env)
- Agents: >80% (spymasters, guessers)
- Wrappers: >80% (single_agent_wrapper)
- Experiments: >85% (trackers, experiment runner)

## Design Principles

### Separation of Concerns
- **Core**: Representation-agnostic game logic
- **Views**: Tile ID ↔ representation mapping
- **Envs**: Multi-agent orchestration
- **Agents**: Policy implementations

### Batching Throughout
All environments and agents support `batch_size > 1` for efficient parallel execution.

### Configurable Rewards
Environments accept optional `reward_fn` parameter. Default is sparse (win=+1, loss=-1), but any custom reward function can be provided.

### Reality Layer as Optional
Can be toggled on/off. When enabled, preserves combinatorial structure while allowing continuous policies.

## Common Patterns

### Agent Action Format

```python
# Spymaster (word-based)
action = {"clue": ["FIRE"], "clue_number": torch.tensor([2], dtype=torch.int32)}

# Spymaster (vector-based)
action = {
    "clue_vec": torch.tensor([[...], [...]]),  # [B, D]
    "clue_number": torch.tensor([2, 3], dtype=torch.int32)  # [B]
}

# Guesser (index-based)
action = {"tile_index": torch.tensor([5], dtype=torch.int32)}  # [B]

# Guesser (word-based - WordEnv only)
action = {"word": ["TREE"]}
```

**Note**: Actions accept both `torch.Tensor` and `numpy.ndarray` (auto-converted), but torch tensors are recommended for zero-copy performance.

### Observation Structure

```python
obs = {
    # Common (all torch.Tensor)
    "revealed": torch.Tensor([B, N], dtype=torch.bool),
    "role_encoding": torch.Tensor([B, 4]),  # [is_red, is_blue, is_spy, is_guesser]
    "current_team": torch.Tensor([B], dtype=torch.int32),
    "phase": torch.Tensor([B], dtype=torch.int32),

    # Spymaster-specific
    "colors": torch.Tensor([B, N], dtype=torch.int32),  # None for guessers

    # Guesser-specific
    "current_clue": List[str],  # WordEnv only
    "current_clue_vec": torch.Tensor([B, D]),  # VectorEnv only
    "current_clue_number": torch.Tensor([B]),
    "remaining_guesses": torch.Tensor([B]),
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{codenames_bot,
  title = {Codenames Bot: Multi-Agent Environment for Deep RL and LLM Research},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/codenames-bot}
}
```

## Acknowledgments

- Built on top of the Codenames board game by Vlaada Chvátil
- Uses sentence-transformers for semantic embedding agents
- Inspired by OpenAI Gym and PettingZoo multi-agent APIs
