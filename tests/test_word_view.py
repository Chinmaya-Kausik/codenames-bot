"""
Tests for views.word_view module.
"""

import pytest

from views.word_view import WordView


def test_word_view_initialization():
    """Test basic initialization."""
    words = ["apple", "banana", "cherry"]
    view = WordView(words)

    assert view.board_size == 3
    assert view.words == ["APPLE", "BANANA", "CHERRY"]  # Uppercased


def test_get_word():
    """Test getting word by index."""
    words = ["apple", "banana", "cherry"]
    view = WordView(words)

    assert view.get_word(0) == "APPLE"
    assert view.get_word(1) == "BANANA"
    assert view.get_word(2) == "CHERRY"

    # Test out of range
    with pytest.raises(IndexError):
        view.get_word(3)


def test_get_index():
    """Test getting index by word."""
    words = ["apple", "banana", "cherry"]
    view = WordView(words)

    assert view.get_index("apple") == 0
    assert view.get_index("APPLE") == 0  # Case insensitive
    assert view.get_index("Banana") == 1
    assert view.get_index("CHERRY") == 2

    # Test not found
    with pytest.raises(ValueError):
        view.get_index("grape")


def test_get_all_words():
    """Test getting all words."""
    words = ["apple", "banana", "cherry"]
    view = WordView(words)

    all_words = view.get_all_words()

    assert all_words == ["APPLE", "BANANA", "CHERRY"]

    # Check it's a copy
    all_words[0] = "MODIFIED"
    assert view.words[0] == "APPLE"


def test_get_unrevealed_words():
    """Test getting unrevealed words."""
    words = ["apple", "banana", "cherry", "date"]
    view = WordView(words)

    revealed = [False, True, False, True]
    unrevealed = view.get_unrevealed_words(revealed)

    assert unrevealed == ["APPLE", "CHERRY"]


def test_create_random():
    """Test creating random word view."""
    word_pool = ["apple", "banana", "cherry", "date", "elderberry", "fig"]

    view = WordView.create_random(word_pool, board_size=3, seed=42)

    assert view.board_size == 3
    assert len(view.words) == 3

    # All words should be from pool
    for word in view.words:
        assert word.upper() in [w.upper() for w in word_pool]

    # Same seed should give same result
    view2 = WordView.create_random(word_pool, board_size=3, seed=42)
    assert view.words == view2.words

    # Different seed should give different result
    view3 = WordView.create_random(word_pool, board_size=3, seed=99)
    assert view.words != view3.words


def test_create_random_insufficient_pool():
    """Test error when word pool is too small."""
    word_pool = ["apple", "banana"]

    with pytest.raises(ValueError):
        WordView.create_random(word_pool, board_size=5, seed=42)


def test_create_batched():
    """Test creating batched word views."""
    word_pool = [f"word_{i}" for i in range(100)]

    views = WordView.create_batched(
        word_pool, batch_size=5, board_size=10, seed=42
    )

    assert len(views) == 5

    for view in views:
        assert isinstance(view, WordView)
        assert view.board_size == 10

    # Views should be different from each other
    assert views[0].words != views[1].words


def test_case_insensitivity():
    """Test that word view is case insensitive."""
    words = ["ApPlE", "BaNaNa", "CHERRY"]
    view = WordView(words)

    # All stored as uppercase
    assert view.words == ["APPLE", "BANANA", "CHERRY"]

    # Lookup is case insensitive
    assert view.get_index("apple") == 0
    assert view.get_index("APPLE") == 0
    assert view.get_index("ApPlE") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
