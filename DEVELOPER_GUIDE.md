# Developer Guide

**Comprehensive guide for developers working on the Codenames Bot codebase.**

## Table of Contents
- [Getting Started](#getting-started)
- [Repository Architecture](#repository-architecture)
- [Core Components](#core-components)
- [Development Workflow](#development-workflow)
- [Adding New Features](#adding-new-features)
- [Testing Guidelines](#testing-guidelines)
- [Performance Optimization](#performance-optimization)
- [Common Patterns](#common-patterns)
- [Debugging Tips](#debugging-tips)
- [Code Style](#code-style)
- [Release Process](#release-process)

## Getting Started

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/codenames-bot.git
cd codenames-bot

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies (including dev tools)
pip install torch numpy pytest sentence-transformers matplotlib pandas seaborn scipy jupyter

# Install in editable mode (for development)
pip install -e .

# Verify setup
python -m pytest tests/ -v
```

### IDE Setup

**VS Code** (Recommended):
```json
// .vscode/settings.json
{
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.analysis.typeCheckingMode": "basic"
}
```

**PyCharm**:
- Mark `codenames-bot/` as Sources Root
- Set pytest as default test runner
- Enable type checking

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Repository Architecture

### Directory Structure

```
codenames-bot/
├── core/                   # 🎯 Core game logic (representation-agnostic)
│   ├── game_state.py      # Central game state with PyTorch tensors
│   ├── clue_vocab.py      # Clue vocabulary for reality layer
│   └── reality_layer.py   # Continuous→discrete conversion layer
│
├── views/                  # 🔍 Representation mappings (tile ID ↔ representation)
│   ├── word_view.py       # Maps tile IDs to word strings
│   └── vector_view.py     # Maps tile IDs to embedding vectors
│
├── envs/                   # 🎮 Multi-agent environments
│   ├── common.py          # Shared environment utilities
│   ├── word_batch_env.py  # Word-based environment (for LLMs/humans)
│   ├── vector_batch_env.py # Vector-based environment (for neural nets)
│   └── wrappers/
│       └── single_agent_wrapper.py  # Wrapper for single-agent RL
│
├── agents/                 # 🤖 Agent implementations
│   ├── spymaster/         # Spymaster agents (give clues)
│   │   ├── base_spymaster.py       # Abstract base class
│   │   ├── random_spymaster.py     # Random baseline
│   │   └── embedding_spymaster.py  # Embedding-based agent
│   └── guesser/           # Guesser agents (select tiles)
│       ├── base_guesser.py         # Abstract base class
│       ├── random_guesser.py       # Random baseline
│       └── embedding_guesser.py    # Embedding-based agent
│
├── experiments/            # 📊 Experiment orchestration
│   ├── trackers.py        # Data collection (Summary, Episode, Trajectory)
│   └── multi_agent_experiment.py  # Experiment runner with sweeps
│
├── utils/                  # 🛠️ Utilities
│   ├── device.py          # GPU/CPU device detection and management
│   ├── embeddings.py      # Sentence embedding utilities
│   ├── wordlist.py        # Codenames word pool management
│   └── notebook_runner.py # Headless notebook execution
│
├── tests/                  # ✅ Comprehensive test suite
│   ├── conftest.py        # Shared pytest fixtures
│   ├── test_*.py          # Unit and integration tests
│   └── ...
│
├── notebooks/              # 📓 Interactive Jupyter demos
│   ├── 01_getting_started.ipynb
│   ├── 02_multi_agent_experiments.ipynb
│   ├── 03_training_single_agent.ipynb
│   └── 04_parameter_sweeps.ipynb
│
├── docs/                   # 📚 Documentation
│   ├── CODE_SIMPLIFICATIONS.md
│   └── MULTI_TEAM_REFACTORING.md
│
├── README.md               # User-facing documentation
├── DEVELOPER_GUIDE.md     # This file
├── AGENTS.md              # Repository guidelines (legacy)
└── benchmark_gpu.py       # GPU performance benchmarking
```

### Design Principles

1. **Separation of Concerns**
   - `core/`: Pure game logic, representation-agnostic
   - `views/`: Handle tile ID ↔ representation mapping only
   - `envs/`: Multi-agent orchestration and observation building
   - `agents/`: Policy implementations

2. **Batching Throughout**
   - All tensors have batch dimension first: `[B, ...]`
   - Environments support `batch_size > 1` natively
   - Agents handle batched observations automatically

3. **PyTorch Everywhere**
   - Game state uses PyTorch tensors (not NumPy)
   - Enables GPU acceleration and zero-copy performance
   - Convert to NumPy only at experiment boundaries (for plotting, saving)

4. **Explicit is Better Than Implicit**
   - Pass `device` explicitly to constructors
   - Use type hints on all public APIs
   - Document tensor shapes in comments: `# [B, N, D]`

## Core Components

### 1. GameState (`core/game_state.py`)

**Purpose**: Represents the state of multiple games in parallel.

**Key Tensors**:
```python
class GameState:
    colors: torch.Tensor           # [B, N] - Tile colors (0=red, 1=blue, 2=neutral, 3=assassin)
    revealed: torch.Tensor         # [B, N] - Whether each tile is revealed
    current_team: torch.Tensor     # [B] - Which team's turn (0=red, 1=blue)
    current_phase: torch.Tensor    # [B] - Phase (0=spymaster, 1=guesser)
    game_over: torch.Tensor        # [B] - Whether game has ended
    winner: torch.Tensor           # [B] - Winner (-1=none, 0=red, 1=blue, 2=assassin hit)
    turn_count: torch.Tensor       # [B] - Number of turns elapsed
    remaining_guesses: torch.Tensor # [B] - Guesses left in current turn
```

**Key Methods**:
- `reset()` - Initialize new games
- `reveal_tile(batch_indices, tile_indices)` - Reveal tiles and update state
- `switch_turn()` - Switch to next team
- `get_active_agent_masks()` - Which agents are currently active

**Adding New State**:
1. Add tensor attribute in `__init__`
2. Initialize in `reset()`
3. Update in relevant methods (`reveal_tile`, etc.)
4. Add tests in `tests/test_game_state.py`

### 2. Views (`views/`)

**Purpose**: Map tile IDs (0-24) to representations (words or vectors).

**WordView** (`views/word_view.py`):
```python
class WordView:
    def __init__(self, batch_size: int, board_size: int, device: str):
        # self.words: List[List[str]] - [B][N] word lists
        pass

    def get_word(self, batch_idx: int, tile_idx: int) -> str:
        """Get word for a specific tile."""
        return self.words[batch_idx][tile_idx]

    def find_word(self, batch_idx: int, word: str) -> int:
        """Find tile index for a word."""
        return self.words[batch_idx].index(word)
```

**VectorView** (`views/vector_view.py`):
```python
class VectorView:
    def __init__(self, batch_size: int, board_size: int, embedding_dim: int, device: str):
        # self.board_vectors: torch.Tensor - [B, N, D] embeddings
        pass

    def get_nearest_vector(self, batch_idx: int, query_vector: torch.Tensor) -> int:
        """Find nearest tile to a query vector."""
        # Compute cosine similarities and return nearest index
        pass
```

**When to Use**:
- Environments use views to build observations
- Never use views for game logic (that's `GameState`'s job)
- Views are stateless - they just map IDs ↔ representations

### 3. Environments (`envs/`)

**Common Interface**:
```python
class BaseEnv:
    agent_ids = ["red_spy", "red_guess", "blue_spy", "blue_guess"]

    def reset(self, seed: Optional[int] = None) -> dict:
        """Reset and return initial observations."""
        pass

    def step(self, actions_dict: dict) -> Tuple[dict, dict, dict, dict]:
        """Execute actions and return (obs, rewards, dones, infos)."""
        pass
```

**Word Batch Environment** (`envs/word_batch_env.py`):
- Uses `WordView` for tile ↔ word mapping
- Clues are strings: `{"clue": ["NATURE"], "clue_number": tensor([2])}`
- Guesses can be by word: `{"word": ["TREE"]}` or index: `{"tile_index": tensor([5])}`

**Vector Batch Environment** (`envs/vector_batch_env.py`):
- Uses `VectorView` for tile ↔ vector mapping
- Clues are vectors: `{"clue_vec": tensor([B, D]), "clue_number": tensor([B])}`
- Guesses are always by index: `{"tile_index": tensor([B])}`
- Optional `reality_layer` for continuous→discrete snapping

**Key Methods**:
- `_process_spymaster_actions()` - Handle spymaster clues
- `_process_guesser_actions()` - Handle guesser tile selections
- `_build_observations()` - Construct role-specific observations
- `_get_rewards()` - Compute rewards (calls custom `reward_fn` if provided)

### 4. Agents (`agents/`)

**Base Classes**:
```python
class BaseSpymaster(ABC):
    def __init__(self, team: str, params: Optional[SpymasterParams] = None):
        self.team = team  # "red" or "blue"
        self.params = params or SpymasterParams()

    @abstractmethod
    def get_clue(self, obs: dict) -> dict:
        """Return clue action: {"clue": [...], "clue_number": tensor(...)}"""
        pass

class BaseGuesser(ABC):
    def __init__(self, team: str, params: Optional[GuesserParams] = None):
        self.team = team
        self.params = params or GuesserParams()

    @abstractmethod
    def get_guess(self, obs: dict) -> dict:
        """Return guess action: {"tile_index": tensor(...)}"""
        pass
```

**Agent Design Pattern**:
1. Inherit from base class (`BaseSpymaster` or `BaseGuesser`)
2. Accept batched observations (handle `batch_size` > 1)
3. Return batched actions (tensors with batch dimension)
4. Use `params` dataclass for configuration
5. Be deterministic given a seed (for reproducibility)

**Example: Adding a New Agent**:
```python
# agents/guesser/nearest_guesser.py
from agents.guesser.base_guesser import BaseGuesser
import torch

class NearestGuesser(BaseGuesser):
    """Always guess tile nearest to clue vector."""

    def get_guess(self, obs: dict) -> dict:
        batch_size = obs['revealed'].shape[0]
        tile_indices = []

        for b in range(batch_size):
            # Get unrevealed tiles
            unrevealed = (~obs['revealed'][b]).nonzero(as_tuple=True)[0]

            if len(unrevealed) > 0:
                # Simple strategy: pick first unrevealed
                tile_indices.append(unrevealed[0].item())
            else:
                tile_indices.append(0)

        return {"tile_index": torch.tensor(tile_indices, dtype=torch.int32)}
```

### 5. Trackers (`experiments/trackers.py`)

**Purpose**: Collect data during game execution via callbacks.

**Base Interface**:
```python
class GameTracker(ABC):
    @abstractmethod
    def on_step(self, step, obs_dict, actions_dict, rewards_dict, dones_dict, infos_dict):
        """Called after each environment step."""
        pass

    @abstractmethod
    def on_episode_end(self, episode_idx, final_infos):
        """Called when a game ends."""
        pass

    @abstractmethod
    def get_results(self):
        """Return accumulated results."""
        pass
```

**Built-in Trackers**:
- `SummaryTracker` - O(1) memory, computes aggregate stats
- `EpisodeTracker` - O(n_games) memory, stores per-episode summaries
- `TrajectoryTracker` - O(n_games × n_steps) memory, stores full trajectories

**Batch Handling**:
- All trackers inherit from `BaseBatchTracker` for batch utilities
- Use `_to_numpy()` helper to convert torch → numpy
- Use `_iter_batch_infos()` to iterate over games in a batch

**Adding Custom Tracker**:
```python
from experiments.trackers import BaseBatchTracker, GameTracker

class MyCustomTracker(BaseBatchTracker, GameTracker):
    def __init__(self):
        super().__init__()
        self.my_data = []

    def on_step(self, step, obs_dict, actions_dict, rewards_dict, dones_dict, infos_dict):
        if not self._initialized:
            self._ensure_initialized(rewards_dict)
        # Collect step data...

    def on_episode_end(self, episode_idx, final_infos):
        for winner, turns in self._iter_batch_infos(final_infos):
            # Process each game in batch
            self.my_data.append({"winner": winner, "turns": turns})

    def get_results(self):
        return self.my_data
```

## Development Workflow

### 1. Feature Development

**Process**:
1. Create feature branch: `git checkout -b feature/my-feature`
2. Write tests first (TDD recommended)
3. Implement feature
4. Run tests: `pytest tests/ -v`
5. Update documentation
6. Submit PR

**Example: Adding Dense Rewards**:
```bash
# 1. Create branch
git checkout -b feature/dense-rewards

# 2. Write test
# tests/test_word_batch_env.py
def test_dense_rewards():
    def dense_reward(prev_state, new_state, agent_id, team_idx):
        # ... reward logic
        pass

    env = WordBatchEnv(batch_size=4, reward_fn=dense_reward)
    # ... test that rewards are computed correctly

# 3. Run test (should fail)
pytest tests/test_word_batch_env.py::test_dense_rewards -v

# 4. Implement feature (already works with current API!)

# 5. Run test (should pass)
pytest tests/test_word_batch_env.py::test_dense_rewards -v

# 6. Commit and push
git add tests/test_word_batch_env.py
git commit -m "Add test for dense reward functions"
git push origin feature/dense-rewards
```

### 2. Bug Fixes

**Process**:
1. Reproduce bug with a failing test
2. Fix bug
3. Verify test passes
4. Add regression test if needed
5. Document fix in commit message

**Example**:
```python
# tests/test_game_state.py
def test_assassin_ends_game():
    """Regression test for issue #42: Game doesn't end when assassin hit."""
    state = GameState(batch_size=1, board_size=25)
    state.reset()

    # Set assassin at index 0
    state.colors[0, 0] = 3  # ASSASSIN

    # Reveal assassin
    state.reveal_tile(batch_indices=torch.tensor([0]), tile_indices=torch.tensor([0]))

    assert state.game_over[0], "Game should end when assassin revealed"
    assert state.winner[0] == 2, "Winner should be 2 (assassin)"
```

### 3. Performance Optimization

**Profiling**:
```python
# Use torch profiler
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Your code here
    env.step(actions_dict)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Benchmarking**:
```bash
# Run GPU benchmark
python benchmark_gpu.py

# Profile specific code
python -m cProfile -o profile.stats my_script.py
python -m pstats profile.stats
```

**Common Bottlenecks**:
1. CPU↔GPU transfers - Keep data on GPU
2. Small batch sizes - Use batch_size >= 32
3. Python loops over batch - Vectorize with torch operations
4. Unnecessary `.clone()` - Use in-place ops when safe

## Adding New Features

### Adding a New Agent Type

**1. Create base class** (if new role):
```python
# agents/observer/base_observer.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ObserverParams:
    seed: int = 42

class BaseObserver(ABC):
    def __init__(self, team: str, params: Optional[ObserverParams] = None):
        self.team = team
        self.params = params or ObserverParams()

    @abstractmethod
    def observe(self, obs: dict) -> dict:
        """Return observation insight."""
        pass
```

**2. Implement concrete agent**:
```python
# agents/observer/random_observer.py
import numpy as np
from .base_observer import BaseObserver

class RandomObserver(BaseObserver):
    def observe(self, obs: dict) -> dict:
        np.random.seed(self.params.seed)
        # Implementation...
        return {"insight": "random"}
```

**3. Add to `__init__.py`**:
```python
# agents/observer/__init__.py
from .base_observer import BaseObserver, ObserverParams
from .random_observer import RandomObserver

__all__ = ["BaseObserver", "ObserverParams", "RandomObserver"]
```

**4. Write tests**:
```python
# tests/test_observer_agents.py
from agents.observer import RandomObserver

def test_random_observer_basic():
    observer = RandomObserver(team="red")
    # Test basic functionality...
```

### Adding a New Environment

**1. Inherit from base or copy existing**:
```python
# envs/text_batch_env.py
from envs.word_batch_env import WordBatchEnv

class TextBatchEnv(WordBatchEnv):
    """Environment with full text descriptions instead of single words."""

    def _build_observations(self, state: GameState) -> dict:
        obs_dict = super()._build_observations(state)

        # Add text descriptions
        for agent_id in self.agent_ids:
            obs_dict[agent_id]["descriptions"] = self._get_descriptions()

        return obs_dict
```

**2. Add tests**:
```python
# tests/test_text_batch_env.py
def test_text_env_has_descriptions():
    env = TextBatchEnv(batch_size=1)
    obs = env.reset()

    assert "descriptions" in obs["red_spy"]
```

### Adding a New Tracker

**1. Inherit from base classes**:
```python
# experiments/trackers.py
class ClueTracker(BaseBatchTracker, GameTracker):
    """Track all clues given during games."""

    def __init__(self):
        super().__init__()
        self.clues = []

    def on_step(self, step, obs_dict, actions_dict, rewards_dict, dones_dict, infos_dict):
        if not self._initialized:
            self._ensure_initialized(rewards_dict)

        # Extract clues from spymaster actions
        for agent_id in ["red_spy", "blue_spy"]:
            if agent_id in actions_dict and "clue" in actions_dict[agent_id]:
                clues = actions_dict[agent_id]["clue"]
                for clue in clues:
                    self.clues.append({"agent": agent_id, "clue": clue, "step": step})

    def on_episode_end(self, episode_idx, final_infos):
        pass  # No per-episode processing needed

    def get_results(self):
        return self.clues
```

**2. Add to exports**:
```python
# experiments/__init__.py
from .trackers import SummaryTracker, EpisodeTracker, TrajectoryTracker, ClueTracker

__all__ = [..., "ClueTracker"]
```

## Testing Guidelines

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_game_state.py       # Core game state tests
├── test_word_batch_env.py   # WordBatchEnv tests
├── test_vector_batch_env.py # VectorBatchEnv tests
├── test_spymaster_agents.py # Spymaster agent tests
├── test_guesser_agents.py   # Guesser agent tests
└── ...
```

### Fixtures (`conftest.py`)

```python
import pytest
import torch

@pytest.fixture
def device():
    """Get available device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def simple_env(device):
    """Simple environment for testing."""
    from envs.word_batch_env import WordBatchEnv
    return WordBatchEnv(batch_size=2, seed=42, device=device)

@pytest.fixture
def random_agents():
    """Random agents for all roles."""
    from agents.spymaster import RandomSpymaster
    from agents.guesser import RandomGuesser

    return {
        "red_spy": RandomSpymaster(team="red"),
        "red_guess": RandomGuesser(team="red"),
        "blue_spy": RandomSpymaster(team="blue"),
        "blue_guess": RandomGuesser(team="blue"),
    }
```

### Test Patterns

**1. Basic Functionality**:
```python
def test_environment_reset(simple_env):
    """Test environment reset returns valid observations."""
    obs_dict = simple_env.reset()

    assert len(obs_dict) == 4  # Four agents
    assert "red_spy" in obs_dict
    assert "words" in obs_dict["red_spy"]
    assert len(obs_dict["red_spy"]["words"]) == 2  # batch_size=2
```

**2. Determinism**:
```python
def test_environment_deterministic():
    """Test that same seed gives same results."""
    env1 = WordBatchEnv(batch_size=1, seed=42)
    env2 = WordBatchEnv(batch_size=1, seed=42)

    obs1 = env1.reset(seed=42)
    obs2 = env2.reset(seed=42)

    assert obs1["red_spy"]["words"] == obs2["red_spy"]["words"]
```

**3. Edge Cases**:
```python
def test_reveal_last_tile():
    """Test revealing the last unrevealed tile."""
    state = GameState(batch_size=1, board_size=25)
    state.reset()

    # Reveal all but one tile
    for i in range(24):
        state.reveal_tile(torch.tensor([0]), torch.tensor([i]))

    assert not state.game_over[0]

    # Reveal last tile
    state.reveal_tile(torch.tensor([0]), torch.tensor([24]))
    # Game may or may not be over depending on tile colors
```

**4. Batching**:
```python
def test_batched_execution():
    """Test that batch_size > 1 works correctly."""
    env = WordBatchEnv(batch_size=8, seed=42)
    obs_dict = env.reset()

    # All observations should have batch dimension 8
    assert obs_dict["red_spy"]["revealed"].shape[0] == 8
    assert len(obs_dict["red_spy"]["words"]) == 8
```

**5. Torch/NumPy Compatibility**:
```python
def test_actions_accept_numpy():
    """Test that actions can be numpy arrays."""
    import numpy as np

    env = WordBatchEnv(batch_size=2, seed=42)
    env.reset()

    # NumPy action
    action = {"tile_index": np.array([0, 1], dtype=np.int32)}
    obs, rewards, dones, infos = env.step({"red_guess": action, ...})

    # Should work without error
    assert rewards["red_guess"].shape == (2,)
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_word_batch_env.py -v

# Specific test
pytest tests/test_word_batch_env.py::test_environment_reset -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Fast (skip slow tests)
pytest tests/ -m "not slow"

# Only GPU tests
pytest tests/ -m "gpu"

# Parallel execution (faster)
pytest tests/ -n auto
```

### Marking Tests

```python
import pytest

@pytest.mark.slow
def test_long_running_experiment():
    """This test takes >10 seconds."""
    pass

@pytest.mark.gpu
def test_cuda_performance():
    """This test requires GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pass

@pytest.mark.parametrize("batch_size", [1, 4, 32])
def test_various_batch_sizes(batch_size):
    """Test with different batch sizes."""
    env = WordBatchEnv(batch_size=batch_size)
    assert env.batch_size == batch_size
```

## Performance Optimization

### GPU Best Practices

**1. Keep Data on GPU**:
```python
# Good: Data stays on GPU
obs = env.reset()  # Already on GPU
action = model(obs)  # Computed on GPU
obs, reward, done, info = env.step(action)  # Stays on GPU

# Bad: Unnecessary transfers
obs = env.reset()
obs_cpu = {k: v.cpu().numpy() for k, v in obs.items()}  # ❌ Don't do this
action = process_on_cpu(obs_cpu)  # ❌ Processing on CPU
action_gpu = torch.tensor(action).to("cuda")  # ❌ Transfer back
```

**2. Batch Operations**:
```python
# Good: Vectorized
revealed_count = state.revealed.sum(dim=1)  # [B]

# Bad: Python loop
revealed_count = torch.tensor([state.revealed[b].sum() for b in range(batch_size)])
```

**3. In-Place Operations**:
```python
# Good: In-place (when safe)
state.revealed[indices] = True  # No new tensor allocated

# Bad: Creates new tensor
state.revealed = state.revealed | new_mask  # Allocates new tensor
```

**4. Avoid Small Batches**:
```python
# Good: Batch size 32+
env = WordBatchEnv(batch_size=128)  # GPU utilization: 90%

# Bad: Small batch
env = WordBatchEnv(batch_size=1)  # GPU utilization: 5%
```

### Memory Optimization

**1. Clear Unused Tensors**:
```python
# Free memory after large computation
large_tensor = compute_something()
result = large_tensor.sum()
del large_tensor  # Free memory
torch.cuda.empty_cache()  # If on CUDA
```

**2. Use Appropriate Data Types**:
```python
# Good: bool for binary data
revealed = torch.zeros(batch_size, 25, dtype=torch.bool)

# Bad: float32 uses 4x more memory
revealed = torch.zeros(batch_size, 25, dtype=torch.float32)
```

**3. Gradient Tracking**:
```python
# Disable gradients for inference
with torch.no_grad():
    action = model(obs)  # No gradient computation
```

### Profiling Tools

**torch.profiler**:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(100):
        env.step(actions_dict)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**cProfile**:
```bash
python -m cProfile -o profile.stats train.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

**Memory Profiling**:
```python
import torch

# Track memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Common Patterns

### 1. Handling Batched Observations

```python
def process_observation(obs: dict) -> torch.Tensor:
    """Process batched observation into features."""
    batch_size = obs['revealed'].shape[0]

    # Extract and concatenate features
    revealed = obs['revealed'].float()  # [B, 25]
    clue_num = obs['current_clue_number'].unsqueeze(-1)  # [B, 1]

    features = torch.cat([revealed, clue_num], dim=-1)  # [B, 26]
    return features
```

### 2. Custom Reward Functions

**Default rewards: Dense** (good for RL training)
```python
from envs.word_batch_env import default_dense_reward, default_sparse_reward

# Use sparse rewards instead of default dense
env = WordBatchEnv(batch_size=8, reward_fn=default_sparse_reward)
```

**Custom reward function example:**
```python
import torch
from core.game_state import GameState

def my_reward_fn(prev_state: GameState, new_state: GameState,
                 agent_id: str, team_idx: int) -> torch.Tensor:
    """
    Custom reward function - must be batched!

    Args:
        prev_state: GameState before action
        new_state: GameState after action
        agent_id: Agent ID (e.g., "red_spy")
        team_idx: Team index (0=red, 1=blue)

    Returns:
        Tensor[B] of rewards for each game in batch
    """
    batch_size = new_state.game_over.shape[0]
    device = new_state.game_over.device
    rewards = torch.zeros(batch_size, device=device)

    # Game end rewards
    newly_finished = ~prev_state.game_over & new_state.game_over
    won_games = newly_finished & (new_state.winner == team_idx)
    lost_games = newly_finished & (new_state.winner != team_idx) & (new_state.winner >= 0)

    rewards[won_games] = 10.0
    rewards[lost_games] = -10.0

    # Dense shaping: reward for revealing team tiles
    revealed_diff = new_state.revealed & ~prev_state.revealed
    team_tiles = (new_state.colors == team_idx) & revealed_diff
    rewards += team_tiles.sum(dim=1).float() * 0.1  # Small reward per tile

    return rewards

# Use in environment
env = WordBatchEnv(batch_size=8, reward_fn=my_reward_fn)
```

### 3. Agent Parameter Sweeps

```python
from dataclasses import dataclass

@dataclass
class MyAgentParams:
    learning_rate: float = 0.001
    hidden_dim: int = 128
    n_layers: int = 2

# Sweep over parameters
param_grid = [
    MyAgentParams(lr=0.001, hidden_dim=64, n_layers=2),
    MyAgentParams(lr=0.01, hidden_dim=128, n_layers=3),
    # ...
]

for params in param_grid:
    agent = MyAgent(params=params)
    # Train and evaluate
```

### 4. Device-Agnostic Code

```python
from utils.device import get_device

class MyAgent:
    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self.model = MyModel().to(self.device)

    def get_action(self, obs: dict) -> dict:
        # Ensure obs is on correct device
        obs_tensor = self._obs_to_tensor(obs).to(self.device)

        # Compute action
        with torch.no_grad():
            action = self.model(obs_tensor)

        return {"tile_index": action}
```

### 5. Tracker for Custom Metrics

```python
class WinRateByTurns(BaseBatchTracker, GameTracker):
    """Track win rate bucketed by game length."""

    def __init__(self, bucket_size: int = 5):
        super().__init__()
        self.bucket_size = bucket_size
        self.buckets = {}  # turn_bucket -> {"red_wins": 0, "blue_wins": 0, "games": 0}

    def on_step(self, ...):
        if not self._initialized:
            self._ensure_initialized(rewards_dict)

    def on_episode_end(self, episode_idx, final_infos):
        if not self._initialized:
            self._ensure_initialized(final_infos)

        for winner, turns in self._iter_batch_infos(final_infos):
            bucket = (turns // self.bucket_size) * self.bucket_size

            if bucket not in self.buckets:
                self.buckets[bucket] = {"red_wins": 0, "blue_wins": 0, "games": 0}

            self.buckets[bucket]["games"] += 1
            if winner == 0:
                self.buckets[bucket]["red_wins"] += 1
            elif winner == 1:
                self.buckets[bucket]["blue_wins"] += 1

    def get_results(self):
        results = {}
        for bucket, stats in self.buckets.items():
            if stats["games"] > 0:
                results[f"{bucket}-{bucket + self.bucket_size}"] = {
                    "red_wr": stats["red_wins"] / stats["games"],
                    "blue_wr": stats["blue_wins"] / stats["games"],
                    "games": stats["games"],
                }
        return results
```

## Debugging Tips

### 1. Tensor Shape Debugging

```python
def debug_shapes(tensor_dict: dict, prefix: str = ""):
    """Print shapes of all tensors in a dict."""
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{prefix}{key}: {value.shape}")
        elif isinstance(value, dict):
            debug_shapes(value, prefix=f"{prefix}{key}.")
        elif isinstance(value, list) and value and isinstance(value[0], str):
            print(f"{prefix}{key}: List[str] len={len(value)}")

# Usage
obs_dict = env.reset()
debug_shapes(obs_dict, prefix="obs.")
```

### 2. Visualizing Game State

```python
def print_board(state: GameState, batch_idx: int = 0):
    """Pretty-print game board."""
    color_map = {0: "🔴", 1: "🔵", 2: "⬜", 3: "💀"}
    revealed_map = {True: "✓", False: " "}

    print("\nBoard:")
    for i in range(5):
        row = []
        for j in range(5):
            idx = i * 5 + j
            color = color_map[state.colors[batch_idx, idx].item()]
            revealed = revealed_map[state.revealed[batch_idx, idx].item()]
            row.append(f"{color}{revealed}")
        print(" ".join(row))
```

### 3. Tracking NaNs and Infs

```python
def check_tensors(tensor_dict: dict, name: str = "tensor"):
    """Check for NaN or Inf in tensors."""
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                raise ValueError(f"NaN detected in {name}.{key}")
            if torch.isinf(value).any():
                raise ValueError(f"Inf detected in {name}.{key}")

# Use during development
rewards_dict = env.step(actions_dict)[1]
check_tensors(rewards_dict, name="rewards")
```

### 4. Determinism Debugging

```python
def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use at start of script
set_all_seeds(42)
```

### 5. Interactive Debugging

```python
# Use IPython for post-mortem debugging
import IPython

try:
    # Your code
    env.step(actions_dict)
except Exception:
    IPython.embed()  # Drop into interactive shell on error

# Or use pdb
import pdb; pdb.set_trace()  # Breakpoint
```

## Code Style

### Python Style Guide

**Follow PEP 8** with these additions:

1. **Type Hints**: Use on all public APIs
   ```python
   def process_clue(clue: str, count: int) -> List[str]:
       pass
   ```

2. **Docstrings**: Google style
   ```python
   def my_function(arg1: int, arg2: str) -> bool:
       """
       Short description.

       Longer description if needed.

       Args:
           arg1: Description of arg1
           arg2: Description of arg2

       Returns:
           Description of return value

       Raises:
           ValueError: If arg1 is negative
       """
       pass
   ```

3. **Imports**: Sorted (stdlib, third-party, local)
   ```python
   # Standard library
   import os
   from typing import List, Optional

   # Third-party
   import numpy as np
   import torch

   # Local
   from core.game_state import GameState
   from utils.device import get_device
   ```

4. **Line Length**: Max 100 characters (not 79)

5. **Naming**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
   - Private: `_leading_underscore`

### Tensor Shape Comments

**Always document tensor shapes**:
```python
# Good
board_vectors = torch.randn(batch_size, board_size, embedding_dim)  # [B, N, D]

# Also good - inline
def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, D] -> [B, N]
    pass
```

### Device Handling

**Always accept device parameter**:
```python
class MyClass:
    def __init__(self, batch_size: int, device: str = "cpu"):
        self.device = device
        self.tensor = torch.zeros(batch_size, device=device)
```

### Error Messages

**Be specific and actionable**:
```python
# Good
raise ValueError(
    f"batch_size must be positive, got {batch_size}. "
    "Try batch_size=1 for single game."
)

# Bad
raise ValueError("Invalid batch size")
```

## Release Process

### Version Numbering

**Semantic Versioning** (MAJOR.MINOR.PATCH):
- MAJOR: Breaking API changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes

### Checklist

**Before Release**:
- [ ] All tests pass
- [ ] Update CHANGELOG.md
- [ ] Update version in `__init__.py`
- [ ] Update README if API changed
- [ ] Run notebooks end-to-end
- [ ] GPU benchmark results documented
- [ ] Tag release: `git tag v1.2.3`

**Release Notes Template**:
```markdown
## v1.2.3 (2024-12-01)

### Added
- New `DenseRewardTracker` for reward analysis
- Support for custom win conditions

### Changed
- Improved GPU memory usage by 20%
- Updated notebooks for clarity

### Fixed
- Bug in `EpisodeTracker` with variable batch sizes (#42)
- Determinism issue with embedding agents (#45)

### Performance
- 15% faster with batch_size=128 on CUDA

### Breaking Changes
- None
```

---

## Getting Help

- **Documentation**: README.md, this guide, docstrings
- **Examples**: notebooks/, tests/
- **Issues**: GitHub issues
- **Discussions**: GitHub discussions

## Contributing

See main [README.md](README.md#contributing) for contribution guidelines.

**Questions?** Open an issue or discussion on GitHub!

---

**Happy Coding!** 🚀
