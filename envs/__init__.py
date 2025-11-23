"""
Environments module for Codenames games.

This module provides multi-agent environment implementations for both
word-based and vector-based Codenames games with batched execution support.
"""

from envs.word_batch_env import WordBatchEnv
from envs.vector_batch_env import VectorBatchEnv

__all__ = ["WordBatchEnv", "VectorBatchEnv"]
