"""
Agents package for Codenames.

This package provides base classes and implementations for Codenames agents,
separated into spymaster (clue-giving) and guesser (tile-selecting) roles.

Modules:
    spymaster: Spymaster agent implementations (RandomSpymaster, EmbeddingSpymaster)
    guesser: Guesser agent implementations (RandomGuesser, EmbeddingGuesser)
"""

from agents.spymaster import (
    BaseSpymaster,
    RandomSpymaster,
    EmbeddingSpymaster,
    SpymasterParams
)
from agents.guesser import (
    BaseGuesser,
    RandomGuesser,
    EmbeddingGuesser,
    GuesserParams
)

__all__ = [
    "BaseSpymaster",
    "RandomSpymaster",
    "EmbeddingSpymaster",
    "SpymasterParams",
    "BaseGuesser",
    "RandomGuesser",
    "EmbeddingGuesser",
    "GuesserParams",
]
