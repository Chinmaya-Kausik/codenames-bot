# Code Simplifications – New Instruction Set

This replaces the previous guidance. Use it as the definitive checklist for the next round of fixes. Each section lists the issue, required changes, and tests to run.

---

## 1. Word Guess Handling Still Unsafe

**Issue:** In `envs/word_batch_env.py` we now uppercase guesses but still quietly fall back to tile `0` when a word is missing. That means any typo reveals the top-left card with no warning.

**Actions:**
1. Replace the `.get(word.upper(), 0)` lookups with a safe helper that:
   - Uppercases the input.
   - Returns a boolean/None if the word does not exist.
2. When the word is missing, either raise a `ValueError` or skip the guess (e.g., keep the previous tile index and rely on `remaining_guesses` to end the turn). Do **not** reveal tile 0 silently.
3. Add tests where a lowercase valid word succeeds and an invalid word raises or leaves the board untouched.

---

## 2. Observations Still Share Mutable Lists

**Issue:** `_get_observations` now clones tensors but still reuses list references (`words`, `current_clue_words`). Agents that mutate these lists corrupt the environment/global state.

**Actions:**
1. Deep-copy `words` and `current_clue` for each agent before returning (e.g., `[list(words) for words in base["words"]]` for each obs).
2. Alternatively, wrap lists in tuples so they are immutable (`tuple(view.get_all_words())`).
3. Add a regression test that mutates an observation list and asserts the underlying env state stays unchanged.

---

## 3. Spymaster Helper Overwrites Finished Games

**Issue:** `process_spymaster_actions` writes zero vectors/words even for games where neither spymaster is active (e.g., game already finished). This wipes out historical clues shown to guessers.

**Actions:**
1. Accept the previous clue vectors/words/current numbers as optional parameters.
2. When both `active_red` and `active_blue` are false for a slot, return the previous values instead of zeros.
3. Update both envs to pass the current clue data into the helper.
4. Add tests that finish a game and confirm `current_clue_*` remains unchanged.

---

## 4. Reality-Layer Helper’s Shape Guard

**Issue:** The helper now replaces any malformed tensor with zeros. That hides bugs in policy outputs and makes debugging difficult.

**Actions:**
1. Instead of silently zeroing when `vec.shape` doesn’t match, raise a descriptive `ValueError` that includes the offending agent and expected shape.
2. Update tests to cover list/numpy inputs (which should be coerced correctly) and truly malformed shapes (which should raise).

---

## 5. Embedding Cache Device Mixups

**Issue:** `utils.embeddings.encode_vocab` and agent-level caches now store tensors per device, but the cached tensor is moved in-place when a new device requests it. That means CPU callers get a CUDA tensor, etc.

**Actions:**
1. When retrieving from `_GLOBAL_VOCAB_CACHE`, clone to the requested device instead of moving the cached tensor.
2. Apply the same rule for the spymaster/guesser caches: store per-device copies or clone before returning.
3. Add unit tests that mock two different “devices” (e.g., CPU + fake string) and assert the cache returns tensors on the correct device each time.

---

## 6. Experiment Runner Slicing Overhead

**Issue:** `_slice_dict` copies every tensor/list for every step, even though trackers usually only need the environment outputs. This causes significant overhead for large observations.

**Actions:**
1. Limit slicing to the env outputs (`obs_dict`, `rewards_dict`, `dones_dict`, `infos_dict`). Do not slice the policy input `actions_dict`—trackers rarely use it.
2. For `obs_dict`, slice only the tensors/lists you actually need (e.g., top-level arrays). Alternatively, maintain a boolean mask with `games_this_batch` and let trackers use that to index into the arrays.
3. Update `SummaryTracker` tests to confirm they still produce the same statistics after the change.

---

## 7. Word Guess Helper Duplicates WordView Logic

**Issue:** The environment reaches into `WordView._word_to_index`. We already have `WordView.get_index`, which handles casing and raises a clear error.

**Actions:**
1. Replace the raw dict access with `try: idx = self.word_views[b].get_index(word)` / `except ValueError`.
2. Decide on the same “invalid word” strategy as in Section 1 (raise or skip).

---

## 8. Testing Checklist

Run at least:
1. `python -m pytest tests/test_word_batch_env.py tests/test_vector_batch_env.py`
2. `python -m pytest tests/test_multi_agent_experiment.py`
3. Any new tests you add for the cache/device behavior.

Log the test commands and statuses in your PR description.
