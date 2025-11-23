# Repository Guidelines

## Project Structure & Modules
- Core game logic lives in `core/` (game state, clue vocab, reality layer). Views (`views/`) map tile IDs to words or vectors. Environments are in `envs/` with `wrappers/` for single-agent training. Agents split by role under `agents/spymaster/` and `agents/guesser/`. Experiment orchestration is in `experiments/`, utilities in `utils/`, and tests in `tests/`. Interactive demos sit in `notebooks/`; performance benchmarking is `benchmark_gpu.py`.

## Build, Test, and Development Commands
- Install deps (example): `pip install numpy sentence-transformers pytest torch`.
- Run full test suite: `python -m pytest tests -v`.
- Target a module: `python -m pytest tests/test_word_batch_env.py -k word`.
- GPU benchmark: `python benchmark_gpu.py` (uses torch; falls back to CPU).
- Notebook to script: `python utils/notebook_runner.py notebooks/01_getting_started.ipynb` (headless execution).

## Coding Style & Naming
- Python 3 with type hints; prefer explicit types on public APIs.
- 4-space indentation; keep modules import-clean and sorted logically (stdlib → third-party → local).
- Use docstrings for classes/methods; keep public method names descriptive (`get_clue`, `get_guess`).
- Favor NumPy for board logic and PyTorch for embeddings; avoid mixing frameworks in the same function unless necessary.
- Keep random seeds thread-safe; pass `seed` through envs/agents rather than using global RNGs.

## Testing Guidelines
- Framework: `pytest`. Tests live in `tests/test_*.py`; add focused fixtures to `tests/conftest.py` when shared.
- Write deterministic tests (set seeds) and cover both word and vector environments when altering shared logic.
- For new agents, add behavior checks (shape, valid actions) plus minimal property tests (e.g., confidence thresholds).
- Run `python -m pytest tests -v` before opening a PR; include GPU benchmark results if performance is affected.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (`Add vector reality layer fallback`). Group related changes; avoid bundling notebooks unless needed.
- PRs: include summary, testing results, and any benchmark deltas. Link issues when applicable; attach small screenshots for notebook/visual changes.
- Keep notebooks clean (executed cells optional, but avoid large outputs). Do not commit cache or model artifacts (`__pycache__/`, weights).

## Security & Configuration Tips
- Do not store API keys or embedding credentials in code; use environment variables and document required names in the PR.
- Large embeddings or wordlists should be referenced, not checked in; prefer download-on-run patterns inside `utils/`.
