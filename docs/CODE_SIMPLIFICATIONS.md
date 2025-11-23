# Code Simplifications – Mandatory Fix List

Follow every instruction exactly. All decisions have already been made—do not invent alternatives or defer work. Update this file if you discover additional problems.

**STATUS: All fixes completed and verified. All tests passing.**

---

## 1. Blue Guesser Tile Indices

**Problem:** Blue guessers currently ignore `tile_index` actions and default to tile 0, causing immediate losses.

**Exact Steps:**
1. Open `envs/word_batch_env.py`.
2. In `_process_guesser_actions`, mirror the red `tile_index` branch for blue:
   ```python
   elif "tile_index" in blue_guess_actions:
       idx_array = blue_guess_actions["tile_index"]
       if isinstance(idx_array, torch.Tensor):
           blue_indices = idx_array.to(self.device).to(torch.int32)
       else:
           blue_indices = torch.tensor(idx_array, dtype=torch.int32, device=self.device)
   ```
3. Place this `elif` immediately after the blue word-guess block so the control flow matches the red side.
4. Update/extend `tests/test_word_batch_env.py` with a case where only the blue agent supplies `tile_index`; assert the revealed tile matches that index.

---

## 2. Word Guess Conversion For Inactive Agents

**Problem:** Inactive guessers still attempt to convert their word lists, so an invalid word on the idle team raises `ValueError` and aborts the step.

**Exact Steps:**
1. Still in `_process_guesser_actions`, wrap the red loop with:
   ```python
   if "word" in red_guess_actions:
       word_list = red_guess_actions["word"]
       if isinstance(word_list, list):
           for b in range(min(len(word_list), self.batch_size)):
               if not active_masks["red_guess"][b]:
                   continue
               red_indices[b] = self._word_to_index(b, word_list[b])
   ```
2. Apply the same guard to the blue word loop (`if not active_masks["blue_guess"][b]: continue`).
3. Add a test where the inactive team submits an invalid word while the active team plays normally; ensure no exception is raised and the active guess proceeds.

---

## 3. Tracker Episode Infos When Loop Doesn’t Run

**Problem:** If `max_turns=0` or all games end before the loop runs, `sliced_infos` remains a dict of empty dicts, so `tracker.on_episode_end` crashes when it expects `winner`/`turn_count`.

**Exact Steps:**
1. In `experiments/multi_agent_experiment.py`, move `sliced_infos` initialization to just after `env.reset`:
   ```python
   _, _, _, infos_dict = env.step({agent: env._noop_action(agent) for agent in env.agent_ids})
   ```
   but since we don’t have a noop, instead set `sliced_infos = self._slice_dict(infos_dict, games_this_batch)` immediately after the first call to `env.step`. To avoid extra steps, do this:
   - Initialize `sliced_infos = None` before the loop.
   - Inside the loop, after `infos_dict = env.step(...)`, set `sliced_infos = self._slice_dict(infos_dict, games_this_batch)`.
   - After the loop, if `sliced_infos is None` (meaning no step ran), compute it once using the last known `infos_dict` (the one from `env.reset`, which includes initial winners/turn counts). You can get it by calling a new helper `env.get_infos_initial()` or, easier, by setting `infos_dict = env._get_infos()` immediately after reset and slicing that.
2. Update `tracker.on_episode_end` call to use the final `sliced_infos` computed as above.
3. Add a regression test in `tests/test_multi_agent_experiment.py` with `max_turns=0` to ensure `run_games` completes without errors.

---

## 4. Testing Checklist

Run the following after applying all fixes:
1. `python -m pytest tests/test_word_batch_env.py`
2. `python -m pytest tests/test_multi_agent_experiment.py`
3. Any additional regression tests you added.
Record the commands and results in your PR description.

---

## Implementation Summary

All fixes have been successfully implemented and tested:

### 1. Blue Guesser Tile Indices ✓
- **File**: `envs/word_batch_env.py:316-321`
- **Change**: Added `elif "tile_index" in blue_guess_actions:` branch mirroring red team logic
- **Test**: `tests/test_word_batch_env.py::test_blue_guesser_tile_index` - PASSED

### 2. Word Guess Conversion For Inactive Agents ✓
- **File**: `envs/word_batch_env.py:300-302, 315-317`
- **Change**: Added `if not active_masks["red_guess"][b]: continue` and `if not active_masks["blue_guess"][b]: continue` guards
- **Test**: `tests/test_word_batch_env.py::test_inactive_team_invalid_word_no_error` - PASSED

### 3. Tracker Episode Infos When Loop Doesn't Run ✓
- **File**: `experiments/multi_agent_experiment.py:167, 200-204`
- **Change**: Initialize `sliced_infos = None` before loop, populate from `env._get_infos()` if still None after loop
- **Additional Fix**: Updated all trackers (SummaryTracker, EpisodeTracker, TrajectoryTracker) in `experiments/trackers.py` to initialize agent_ids in `on_episode_end` when max_turns=0
- **Test**: `tests/test_multi_agent_experiment.py::test_max_turns_zero` - PASSED

### Test Results
```
tests/test_word_batch_env.py: 23 passed in 1.00s
tests/test_multi_agent_experiment.py: 20 passed in 12.83s
```

All tests passing. Ready for review.
