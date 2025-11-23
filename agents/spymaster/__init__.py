"""
Spymaster agents for multi-agent Codenames.

This module provides spymaster agent implementations that work with
the multi-agent environment API.
"""

from agents.spymaster.base_spymaster import BaseSpymaster, SpymasterParams
from agents.spymaster.random_spymaster import RandomSpymaster
from agents.spymaster.embedding_spymaster import EmbeddingSpymaster

__all__ = [
    "BaseSpymaster",
    "SpymasterParams",
    "RandomSpymaster",
    "EmbeddingSpymaster",
]
