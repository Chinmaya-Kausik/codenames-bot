"""
Core game logic for Codenames.

This module provides representation-agnostic, batched game logic that can be
used by both word-based and vector-based environments.
"""

from core.game_state import GameState
from core.clue_vocab import ClueVocab
from core.reality_layer import RealityLayer

__all__ = ["GameState", "ClueVocab", "RealityLayer"]
