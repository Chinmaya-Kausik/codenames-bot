"""
Pytest configuration for the Codenames bot.

Ensures tests run against the lightweight hash-based embeddings to avoid
pulling large transformer models or relying on network access.
"""

import os

# Force deterministic fake embeddings for fast, offline test runs.
os.environ.setdefault("CODENAMES_FAKE_EMBEDDINGS", "1")
