# Codenames Bot

**Multi-agent Codenames environment for deep reinforcement learning and LLM research.**

[![Tests](https://img.shields.io/badge/tests-128%20passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.8%2B-blue)]() [![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)]()

## Table of Contents
- [What is This?](#what-is-this)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Examples](#examples)
- [Notebooks](#notebooks)
- [Architecture](#architecture)
- [GPU Acceleration](#gpu-acceleration)
- [Testing](#testing)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## What is This?

This repository provides a complete multi-agent implementation of the board game [Codenames](https://en.wikipedia.org/wiki/Codenames_(board_game)), designed specifically for AI research. Whether you're training deep RL agents, building LLM-based players, or researching multi-agent coordination, this environment provides the tools you need.

### What is Codenames?

Codenames is a word-guessing game where two teams compete:
- **Spymasters** give one-word clues to help their team identify specific words on a 5×5 grid
- **Guessers** try to identify their team's words based on the clue
- **Goal**: Be the first team to reveal all your words, while avoiding the assassin word (instant loss!)

**Example Turn:**
```
Board: [TREE, RIVER, MOON, BANK, FIRE, ...]
Red Spymaster (knows TREE and RIVER are red): "NATURE 2"
Red Guesser: *selects TREE* ✓ *selects RIVER* ✓
```

### Why This Implementation?

✨ **For Researchers:**
- Batched execution: Train on 1000s of parallel games
- GPU acceleration: Full PyTorch implementation
- Flexible rewards: Customize for your training objective
- Multiple representations: Words (LLMs) or vectors (neural nets)

✨ **For Developers:**
- Clean multi-agent API: Standard gym-like interface
- 128 passing tests: Reliable and well-tested
- Comprehensive docs: Examples, notebooks, and guides
- Extensible: Easy to add custom agents and rewards

## Key Features

| Feature | Description |
|---------|-------------|
| 🎮 **Multi-Agent API** | Standard interface with 4 agents per game (2 spymasters + 2 guessers) |
| 🔄 **Dual Environments** | `WordBatchEnv` (LLMs) and `VectorBatchEnv` (neural networks) |
| ⚡ **GPU Accelerated** | Full PyTorch with CUDA/MPS/CPU support |
| 🚀 **Batched Execution** | Run 128 games in parallel, 18.9x faster than single-game |
| 🎯 **Custom Rewards** | Plug in your own reward functions |
| 📊 **Experiment Framework** | Built-in tracking, parameter sweeps, and visualization |
| 🧪 **Well-Tested** | 128 tests with >85% coverage across all modules |
| 📓 **Interactive Demos** | 4 Jupyter notebooks to get started |

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for NVIDIA GPUs)
- MPS (automatic, for Apple Silicon)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/codenames-bot.git
cd codenames-bot

# Install core dependencies
pip install torch numpy pytest

# Optional: Install for embedding-based agents (recommended)
pip install sentence-transformers

# Verify installation
python -m pytest tests/ -v
```

### Development Installation

```bash
# Install with dev dependencies
pip install torch numpy pytest sentence-transformers matplotlib pandas seaborn scipy jupyter

# Run tests
python -m pytest tests/ -v

# Run GPU benchmark
python benchmark_gpu.py
```

## Quick Start

### 30-Second Example

```python
from envs.word_batch_env import WordBatchEnv
from agents.spymaster import RandomSpymaster
from agents.guesser import RandomGuesser

# Create environment (1 game)
env = WordBatchEnv(batch_size=1, seed=42)

# Create agents for both teams
agents = {
    "red_spy": RandomSpymaster(team="red"),
    "red_guess": RandomGuesser(team="red"),
    "blue_spy": RandomSpymaster(team="blue"),
    "blue_guess": RandomGuesser(team="blue"),
}

# Play a game!
obs_dict = env.reset()
for turn in range(50):
    if env.game_state.game_over[0]:
        break

    # Get actions from all agents
    actions = {aid: agent.get_clue(obs_dict[aid]) if 'spy' in aid else agent.get_guess(obs_dict[aid])
               for aid, agent in agents.items()}

    # Step environment
    obs_dict, rewards, dones, infos = env.step(actions)

# Check who won!
winner = infos['red_spy']['winner'][0]
print(f"Winner: {'Red' if winner == 0 else 'Blue' if winner == 1 else 'Draw'}")
```

## Core Concepts

### 1. Environments

Two environment types with identical API:

**WordBatchEnv** - For LLMs and human-readable games
```python
from envs.word_batch_env import WordBatchEnv

env = WordBatchEnv(batch_size=8, seed=42)  # 8 parallel games
obs = env.reset()  # obs['red_spy']['words'] = ["TREE", "RIVER", ...]
```

**VectorBatchEnv** - For neural networks
```python
from envs.vector_batch_env import VectorBatchEnv

env = VectorBatchEnv(batch_size=32, seed=42)  # 32 parallel games
obs = env.reset()  # obs['red_spy']['board_vectors'].shape = [32, 25, 384]
```

### 2. Agents

Four agent IDs per game:
- `red_spy` - Red team spymaster (gives clues)
- `red_guess` - Red team guesser (selects tiles)
- `blue_spy` - Blue team spymaster
- `blue_guess` - Blue team guesser

**Built-in Agents:**
- `RandomSpymaster` / `RandomGuesser` - Random baseline
- `EmbeddingSpymaster` / `EmbeddingGuesser` - Semantic similarity-based

### 3. Observations

Agents receive role-specific observations:

```python
# Spymaster observation
{
    "words": List[List[str]],           # Board words
    "colors": Tensor[B, 25],            # Tile colors (RED=0, BLUE=1, NEUTRAL=2, ASSASSIN=3)
    "revealed": Tensor[B, 25],          # Already revealed tiles
    "role_encoding": Tensor[B, 4],      # [is_red, is_blue, is_spy, is_guesser]
    ...
}

# Guesser observation (no colors!)
{
    "words": List[List[str]],           # Board words
    "revealed": Tensor[B, 25],          # Already revealed tiles
    "current_clue": List[str],          # Current clue from spymaster
    "current_clue_number": Tensor[B],   # Number of words for clue
    "remaining_guesses": Tensor[B],     # Guesses left this turn
    ...
}
```

### 4. Actions

```python
# Spymaster action (give clue)
action = {
    "clue": ["NATURE"],                 # One-word clue
    "clue_number": torch.tensor([2])    # Number of related words
}

# Guesser action (select tile)
action = {
    "tile_index": torch.tensor([5])     # Index 0-24
}
# Or word-based (WordBatchEnv only):
action = {
    "word": ["TREE"]                    # Select by word
}
```

### 5. Rewards

**Default: Dense rewards** (good for RL training)
- +1 per own team tile revealed
- -1 per opponent tile revealed
- -10 for assassin
- +10 for winning, -10 for losing

**Alternative: Sparse rewards** (simpler, but harder to learn)
```python
from envs.word_batch_env import default_sparse_reward

env = WordBatchEnv(batch_size=8, reward_fn=default_sparse_reward)
# Only gives +1 for win, -1 for loss, 0 otherwise
```

**Custom reward functions:**
```python
import torch
from core.game_state import GameState

def my_custom_reward(prev_state: GameState, new_state: GameState,
                     agent_id: str, team_idx: int) -> torch.Tensor:
    """
    Custom reward function - must return tensor of shape [B].

    Args:
        prev_state: GameState before action
        new_state: GameState after action
        agent_id: Agent ID (e.g., "red_spy")
        team_idx: Team index (0=red, 1=blue)

    Returns:
        Tensor[B] of rewards for each game in batch
    """
    batch_size = new_state.game_over.shape[0]
    rewards = torch.zeros(batch_size, device=new_state.game_over.device)

    # Your custom reward logic here
    # Example: Big bonus for winning
    newly_finished = ~prev_state.game_over & new_state.game_over
    won_games = newly_finished & (new_state.winner == team_idx)
    rewards[won_games] = 100.0

    return rewards

env = WordBatchEnv(batch_size=8, reward_fn=my_custom_reward)
```

## Examples

### Example 1: Run 100 Games and Track Win Rates

```python
from experiments import MultiAgentCodenamesExperiment, SummaryTracker
from envs.word_batch_env import WordBatchEnv
from agents.spymaster import RandomSpymaster
from agents.guesser import RandomGuesser

# Create experiment
exp = MultiAgentCodenamesExperiment(
    env_factory=lambda seed: WordBatchEnv(batch_size=8, seed=seed),
    max_turns=50
)

# Define policies
policy_map = {
    "red_spy": lambda obs: RandomSpymaster(team="red").get_clue(obs),
    "red_guess": lambda obs: RandomGuesser(team="red").get_guess(obs),
    "blue_spy": lambda obs: RandomSpymaster(team="blue").get_clue(obs),
    "blue_guess": lambda obs: RandomGuesser(team="blue").get_guess(obs),
}

# Run with tracking
tracker = SummaryTracker()
results = exp.run_games(
    policy_map=policy_map,
    n_games=100,
    tracker=tracker,
    seed=42,
    verbose=True
)

print(f"Red win rate: {results['red_win_rate']:.1%}")
print(f"Blue win rate: {results['blue_win_rate']:.1%}")
print(f"Average game length: {results['avg_turns']:.1f} turns")
```

### Example 2: Train a Single Agent with RL

```python
from envs.wrappers import SingleAgentWrapper
from envs.word_batch_env import WordBatchEnv
import numpy as np

# Wrap environment to focus on training red_guess
env = SingleAgentWrapper(
    env=WordBatchEnv(batch_size=4, seed=42),
    agent_id="red_guess",  # Only control this agent
    policy_map={
        "red_spy": lambda obs: RandomSpymaster(team="red").get_clue(obs),
        "blue_spy": lambda obs: RandomSpymaster(team="blue").get_clue(obs),
        "blue_guess": lambda obs: RandomGuesser(team="blue").get_guess(obs),
    }
)

# Standard RL training loop
obs = env.reset()
for step in range(1000):
    # Your RL policy here
    action = {"tile_index": np.array([np.random.randint(0, 25) for _ in range(4)])}

    obs, reward, done, info = env.step(action)

    # Train your agent with (obs, action, reward, done)
    # ...
```

### Example 3: Parameter Sweep for Agent Optimization

```python
from experiments import MultiAgentCodenamesExperiment, EpisodeTracker
from agents.spymaster import EmbeddingSpymaster, SpymasterParams
from agents.guesser import EmbeddingGuesser, GuesserParams

# Create experiment
exp = MultiAgentCodenamesExperiment(
    env_factory=lambda seed: WordBatchEnv(batch_size=8, seed=seed),
    max_turns=50
)

# Define policy factory
def make_policies(params):
    n_candidates = params['n_candidate_clues']
    risk = params['risk_tolerance']

    return {
        "red_spy": lambda obs: EmbeddingSpymaster(
            team="red",
            params=SpymasterParams(n_candidate_clues=n_candidates, risk_tolerance=risk)
        ).get_clue(obs),
        "red_guess": lambda obs: EmbeddingGuesser(team="red").get_guess(obs),
        "blue_spy": lambda obs: EmbeddingSpymaster(team="blue").get_clue(obs),
        "blue_guess": lambda obs: EmbeddingGuesser(team="blue").get_guess(obs),
    }

# Parameter grid
param_grid = [
    {"n_candidate_clues": 10, "risk_tolerance": 1.0},
    {"n_candidate_clues": 50, "risk_tolerance": 2.0},
    {"n_candidate_clues": 100, "risk_tolerance": 3.0},
]

# Run sweep
results = exp.run_sweep(
    policy_factory=make_policies,
    param_grid=param_grid,
    n_games_per_config=50,
    seed=42
)

# Analyze
for r in results:
    print(f"n_candidates={r['params']['n_candidate_clues']}, "
          f"risk={r['params']['risk_tolerance']}: "
          f"red_wr={r['results']['red_win_rate']:.1%}")
```

## Notebooks

Explore the `notebooks/` directory for interactive tutorials:

| Notebook | Description |
|----------|-------------|
| [01_getting_started.ipynb](notebooks/01_getting_started.ipynb) | Environment basics, agent creation, game loop |
| [02_multi_agent_experiments.ipynb](notebooks/02_multi_agent_experiments.ipynb) | Running experiments with trackers |
| [03_training_single_agent.ipynb](notebooks/03_training_single_agent.ipynb) | Single-agent RL training with wrappers |
| [04_parameter_sweeps.ipynb](notebooks/04_parameter_sweeps.ipynb) | Systematic parameter optimization |

**To run:**
```bash
jupyter notebook notebooks/01_getting_started.ipynb
```

## Architecture

```
codenames-bot/
├── core/                      # 🎯 Core game logic (representation-agnostic)
│   ├── game_state.py         # Batched game state with PyTorch tensors
│   ├── clue_vocab.py         # Clue vocabulary management
│   └── reality_layer.py      # Continuous→discrete conversion
│
├── views/                     # 🔍 Representation mappings
│   ├── word_view.py          # Tile ID ↔ Word string
│   └── vector_view.py        # Tile ID ↔ Embedding vector
│
├── envs/                      # 🎮 Multi-agent environments
│   ├── word_batch_env.py     # Word-based environment (for LLMs)
│   ├── vector_batch_env.py   # Vector-based environment (for neural nets)
│   └── wrappers/
│       └── single_agent_wrapper.py  # Single-agent training wrapper
│
├── agents/                    # 🤖 Agent implementations
│   ├── spymaster/            # Spymaster agents (give clues)
│   │   ├── base_spymaster.py
│   │   ├── random_spymaster.py
│   │   └── embedding_spymaster.py
│   └── guesser/              # Guesser agents (select tiles)
│       ├── base_guesser.py
│       ├── random_guesser.py
│       └── embedding_guesser.py
│
├── experiments/               # 📊 Experiment orchestration
│   ├── trackers.py           # Data collection (Summary, Episode, Trajectory)
│   └── multi_agent_experiment.py  # Experiment runner
│
├── utils/                     # 🛠️ Utilities
│   ├── device.py             # GPU/CPU device management
│   ├── embeddings.py         # Sentence embedding utilities
│   └── wordlist.py           # Codenames word pool
│
├── tests/                     # ✅ Comprehensive test suite (128 tests)
├── notebooks/                 # 📓 Interactive demos
└── docs/                      # 📚 Additional documentation
```

## GPU Acceleration

**All components use PyTorch for zero-copy GPU acceleration:**

| Component | Device | Benefits |
|-----------|--------|----------|
| Game State | GPU | Batched tensor operations |
| Environments | GPU | Parallel game execution |
| Embedding Agents | GPU | Fast similarity computations |
| Views | GPU | Efficient vector lookups |

**Automatic Device Detection:**
```python
from utils.device import get_device

device = get_device()  # Returns 'cuda', 'mps', or 'cpu'
env = WordBatchEnv(batch_size=128, device=device)
```

**Performance (Apple M2 Pro, MPS):**
| Batch Size | Games/sec | Speedup vs B=1 |
|------------|-----------|----------------|
| 1          | 100       | 1.0x           |
| 32         | 1,200     | 12.0x          |
| 128        | 1,892     | 18.9x          |

**Key Insight:** Keep data on GPU during training, only convert to NumPy for analysis.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_word_batch_env.py -v

# Run with coverage report
python -m pytest tests/ --cov=. --cov-report=html

# Run GPU benchmark
python benchmark_gpu.py
```

**Test Coverage:**
- Core modules: 100%
- Environments: >90%
- Agents: >80%
- Experiments: >85%

## Contributing

We welcome contributions! See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for detailed guidelines.

**Quick contribution steps:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

## Troubleshooting

### Common Issues

**1. "sentence-transformers not found"**
```bash
pip install sentence-transformers
```
Note: Only needed for `EmbeddingSpymaster` and `EmbeddingGuesser`. Random agents work without it.

**2. "MPS device not available"**
- Ensure you have macOS 12.3+ with Apple Silicon
- Update PyTorch: `pip install --upgrade torch`

**3. "CUDA out of memory"**
```python
# Reduce batch size
env = WordBatchEnv(batch_size=32)  # Instead of 128
```

**4. "Tests failing on GPU"**
```bash
# Force CPU mode for testing
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v
```

**5. "Notebooks won't run"**
```bash
pip install jupyter matplotlib pandas seaborn scipy
jupyter notebook
```

### Performance Tips

1. **Use larger batch sizes** for better GPU utilization (128+ recommended)
2. **Keep data on GPU** - avoid unnecessary `.cpu()` calls
3. **Use sparse rewards** for faster convergence in RL training
4. **Disable embedding agents** if you just need fast baselines

### Getting Help

- 📖 Read the [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- 🐛 [Open an issue](https://github.com/yourusername/codenames-bot/issues)
- 💬 Check existing issues for solutions

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

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the board game [Codenames](https://en.wikipedia.org/wiki/Codenames_(board_game)) by Vlaada Chvátil
- Uses [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- Inspired by [OpenAI Gym](https://github.com/openai/gym) and [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) multi-agent APIs

---

**Ready to build AI agents that play Codenames?** Start with the [Quick Start](#quick-start) or dive into the [notebooks](notebooks/)! 🚀
