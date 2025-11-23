# Multi-Team Refactoring Analysis

## Executive Summary

**TL;DR:** Supporting N teams requires **moderate refactoring** (30-40% of the codebase). The core architecture is well-positioned for this, but significant changes are needed in game rules, agent IDs, and reward structures.

**Difficulty Rating:** ⭐⭐⭐☆☆ (3/5)
- Easy parts: Color encoding, turn switching, winner tracking
- Hard parts: Game rules design, agent ID generation, reward structure, observations

---

## Current 2-Team Assumptions

### Hardcoded Elements

1. **Agent IDs** (High Impact)
   ```python
   AGENT_IDS = ["red_spy", "red_guess", "blue_spy", "blue_guess"]  # Fixed 4
   ```

2. **Color Constants** (Low Impact)
   ```python
   RED = 0
   BLUE = 1
   NEUTRAL = 2
   ASSASSIN = 3
   ```

3. **Turn Switching** (Low Impact)
   - Alternates between team 0 and team 1
   - Easy to generalize to N teams with round-robin

4. **Tile Distribution** (Medium Impact)
   - Starting team: 9 tiles
   - Other team: 8 tiles
   - Neutral: 7 tiles
   - Assassin: 1 tile
   - How should this scale to N teams?

5. **Win Conditions** (Medium Impact)
   - Win: Reveal all team tiles
   - Loss: Hit assassin or opponent wins
   - With N teams: First to finish? Points-based? Free-for-all?

6. **Reward Structure** (High Impact)
   - Currently: team vs opponent
   - With N teams: Relative ranking? Winner-take-all? Cooperative subsets?

---

## Required Changes by Module

### 1. Core Game State

**File:** `core/game_state.py`

**Current State:**
```python
class GameState:
    colors: np.ndarray  # [B, N] in {0, 1, 2, 3}
    current_team: np.ndarray  # [B] in {0, 1}
```

**Changes Needed:**
```python
class GameState:
    colors: np.ndarray  # [B, N] in {0, 1, ..., n_teams-1, neutral, assassin}
    current_team: np.ndarray  # [B] in {0, 1, ..., n_teams-1}
    n_teams: int  # New parameter
```

**Difficulty:** ⭐⭐☆☆☆ (Easy)
- Most logic is already team-agnostic
- Need to update tile distribution logic
- Need to update win condition checks

### 2. Agent IDs

**File:** `envs/word_batch_env.py`, `envs/vector_batch_env.py`

**Current State:**
```python
AGENT_IDS = ["red_spy", "red_guess", "blue_spy", "blue_guess"]
```

**Changes Needed:**
```python
def generate_agent_ids(n_teams: int, team_names: Optional[List[str]] = None) -> List[str]:
    if team_names is None:
        team_names = [f"team{i}" for i in range(n_teams)]

    agent_ids = []
    for team_name in team_names:
        agent_ids.append(f"{team_name}_spy")
        agent_ids.append(f"{team_name}_guess")

    return agent_ids
```

**Difficulty:** ⭐⭐⭐☆☆ (Medium)
- Easy to generate IDs
- Hard to update all references throughout codebase
- Need to update observation generation, action processing, reward distribution

### 3. Observations

**File:** `envs/word_batch_env.py`, `envs/vector_batch_env.py`

**Current State:**
```python
role_encoding = [is_red, is_blue, is_spy, is_guesser]  # Shape: [B, 4]
```

**Changes Needed:**
```python
# Option A: One-hot team encoding
role_encoding = [*team_one_hot, is_spy, is_guesser]  # Shape: [B, n_teams + 2]

# Option B: Team index + role flags
role_encoding = [team_idx, is_spy, is_guesser]  # Shape: [B, 3]
```

**Difficulty:** ⭐⭐⭐☆☆ (Medium)
- Need to update all observation building
- Neural networks need to handle variable-size team encoding
- Consider: Should networks be team-agnostic?

### 4. Tile Distribution

**Current:** 9-8-7-1 split for 25 tiles

**Questions:**
- **N=3 teams, 25 tiles:** 8-8-8-1? Equal distribution?
- **N=4 teams, 25 tiles:** 6-6-6-6-1? Not enough tiles!
- **Board size flexibility:** Allow larger boards (6×6, 7×7)?
- **Starting team advantage:** Should one team get more tiles?

**Difficulty:** ⭐⭐⭐⭐☆ (Hard - design decision)
- No obvious "correct" generalization
- Affects game balance significantly
- May need playtesting to find good distributions

### 5. Turn Order

**Current:** Alternating (red → blue → red → blue)

**Options for N teams:**
- **Round-robin:** team0 → team1 → team2 → ... → team0
- **Dynamic:** Based on remaining tiles (fewer tiles = more turns?)
- **Random:** Stochastic turn order

**Difficulty:** ⭐⭐☆☆☆ (Easy)
- Round-robin is straightforward
- Just need to update `current_team = (current_team + 1) % n_teams`

### 6. Win Conditions

**Current:**
- Win: Reveal all your team's tiles
- Loss: Hit assassin or opponent reveals all their tiles

**Options for N teams:**

**A. First-to-Finish (Race)**
- First team to reveal all tiles wins
- Others lose
- Assassin: Team that hits it is eliminated; game continues

**B. Points-Based**
- Points for revealing own tiles
- Negative points for opponent tiles, neutrals, assassin
- Highest score at end wins
- Game ends after X turns or all tiles revealed

**C. Last-Team-Standing**
- Hit assassin → eliminated
- Last team remaining wins

**Difficulty:** ⭐⭐⭐⭐☆ (Hard - design decision)
- Each variant has different strategic implications
- Reward shaping becomes more complex
- May want to support multiple win-condition modes

### 7. Rewards

**Current:**
```python
def default_sparse_reward(prev_state, new_state, agent_id, team_idx):
    if new_state.game_over and new_state.winner == team_idx:
        return 1.0
    elif new_state.game_over and new_state.winner != team_idx:
        return -1.0
    return 0.0
```

**Challenges for N teams:**
- **Winner-take-all:** Only 1 team gets +1, rest get -1?
- **Relative ranking:** 1st place: +1, 2nd: 0, 3rd: -0.5, 4th: -1?
- **Zero-sum:** Rewards must sum to 0 across teams?
- **Dense shaping:** Reward for progress relative to other teams?

**Difficulty:** ⭐⭐⭐⭐⭐ (Very Hard - RL research question)
- Multi-agent RL with N>2 is significantly harder
- Credit assignment becomes complex
- May need different reward functions for different training objectives

### 8. Observations - Color Visibility

**Current:**
- Spymasters see all colors
- Guessers see revealed tiles + current clue

**Questions for N teams:**
- Do spymasters see ALL team colors, or only their own + neutral/assassin?
- Partial observability: Each team only knows own tiles?
- Full observability: All spymasters know all tiles (current behavior)?

**Difficulty:** ⭐⭐⭐☆☆ (Medium - design decision)
- Partial observability adds complexity but may be more realistic
- Full observability is simpler but may be unrealistic for N>2

---

## Implementation Strategy

### Phase 1: Parameterize N Teams (Foundation)

**Goal:** Make `n_teams` a parameter throughout codebase

**Changes:**
1. Add `n_teams` parameter to GameState
2. Generalize color constants: `{0, 1, ..., n_teams-1, neutral, assassin}`
3. Update turn switching to round-robin
4. Update agent ID generation
5. Update role encoding in observations

**Estimated Effort:** 2-3 days

**Difficulty:** ⭐⭐☆☆☆

### Phase 2: Tile Distribution & Win Conditions (Game Design)

**Goal:** Define game rules for N>2

**Decisions Needed:**
1. Tile distribution algorithm
   - Equal split? Starting team advantage?
   - Support variable board sizes?
2. Win condition variant
   - First-to-finish? Points-based? Last-standing?
3. Assassin behavior
   - Eliminate team? End game? Multiple assassins?

**Estimated Effort:** 1-2 weeks (including playtesting)

**Difficulty:** ⭐⭐⭐⭐☆

### Phase 3: Reward Engineering (RL Design)

**Goal:** Define reward structures that enable training

**Tasks:**
1. Implement multiple reward variants
2. Test which rewards lead to learning
3. Consider self-play dynamics with N>2
4. Handle credit assignment issues

**Estimated Effort:** 2-4 weeks (research effort)

**Difficulty:** ⭐⭐⭐⭐⭐

### Phase 4: Testing & Validation

**Goal:** Verify N-team implementation works

**Tasks:**
1. Unit tests for N=2, N=3, N=4
2. Integration tests with agents
3. Visual inspection of games
4. Balance testing

**Estimated Effort:** 1 week

**Difficulty:** ⭐⭐⭐☆☆

---

## Backward Compatibility

**Can we maintain N=2 as default and support N>2 as extension?**

**Yes**, with careful design:

```python
class GameState:
    def __init__(self, batch_size, board_size=25, n_teams=2, ...):
        self.n_teams = n_teams
        # ... rest of init
```

All existing code works with `n_teams=2` by default.

**Breaking changes:**
- Agent ID format (but can auto-generate for N=2 as "red", "blue")
- Observation role encoding (but can keep [is_red, is_blue, is_spy, is_guesser] for N=2)

**Strategy:** Feature flag
```python
if self.n_teams == 2:
    # Use legacy behavior
else:
    # Use generalized N-team behavior
```

---

## Open Questions

1. **Board size scaling:** Should board size scale with N teams? (5×5 for N=2, 6×6 for N=3, etc.)?

2. **Clue vocabulary:** With N teams, does ClueVocab size need to increase?

3. **Training complexity:** Can we realistically train policies with N>2? Self-play convergence?

4. **Optimal N:** Is there a sweet spot? N=3 might be interesting, N=5 might be too chaotic.

5. **Hybrid mode:** Support both traditional 2-team and N-team in same codebase, or separate branches?

---

## Recommendation

**Start with N=3 as the first generalization target.**

**Reasons:**
1. N=3 is the minimal extension beyond 2
2. Tile distribution is feasible (8-8-8-1 for 25 tiles)
3. Win conditions have clear semantics (first-to-finish)
4. Not so many teams that it becomes chaotic
5. Validates the architecture without over-engineering

**Roadmap:**
1. ✅ Build solid 2-team multi-agent foundation (current work)
2. 🔄 Add `n_teams` parameter, default N=2
3. 🔄 Implement N=3 mode with first-to-finish win condition
4. 🔄 Test and validate
5. ⏸️ Generalize to arbitrary N (if needed)

---

## Conclusion

Supporting N>2 teams is **feasible but non-trivial**. The current architecture is well-designed for extension, but significant work is needed in:
- Game rule design (tile distribution, win conditions)
- Reward engineering (multi-agent credit assignment)
- Testing and balance

The path forward: **Start with 2-team, design for 3-team extensibility, implement N-team if needed.**

Estimated total effort: **4-6 weeks** for production-ready N-team support.
