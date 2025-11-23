"""
Tests for core.game_state module.
"""

import torch
import pytest

from core.game_state import GameState


def test_game_state_initialization():
    """Test basic initialization of GameState."""
    batch_size = 4
    board_size = 25

    state = GameState(batch_size=batch_size, board_size=board_size, seed=42, device='cpu')

    assert state.batch_size == batch_size
    assert state.board_size == board_size
    assert state.colors.shape == (batch_size, board_size)
    assert state.revealed.shape == (batch_size, board_size)
    assert state.current_team.shape == (batch_size,)
    assert state.phase.shape == (batch_size,)
    assert not torch.any(state.revealed)  # All unrevealed initially
    assert not torch.any(state.game_over)  # No games over initially


def test_color_distribution():
    """Test that color distribution is correct."""
    batch_size = 10
    board_size = 25

    state = GameState(batch_size=batch_size, board_size=board_size, seed=42, device='cpu')

    # Check each board has correct color counts
    for b in range(batch_size):
        colors = state.colors[b]

        # Count colors
        first_team = state.current_team[b].item()
        second_team = 1 - first_team

        first_count = torch.sum(colors == first_team).item()
        second_count = torch.sum(colors == second_team).item()
        neutral_count = torch.sum(colors == GameState.NEUTRAL).item()
        assassin_count = torch.sum(colors == GameState.ASSASSIN).item()

        # Standard 5x5 distribution
        assert first_count == 9
        assert second_count == 8
        assert neutral_count == 7
        assert assassin_count == 1


def test_give_clue():
    """Test clue-giving updates state correctly."""
    batch_size = 2
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    # Both games start in spymaster phase
    assert torch.all(state.phase == GameState.SPYMASTER_PHASE)

    # Give clues
    clue_indices = torch.tensor([5, 10], dtype=torch.int32)
    clue_numbers = torch.tensor([2, 3], dtype=torch.int32)

    state.give_clue(clue_indices, clue_numbers)

    # Check state updated
    assert torch.all(state.phase == GameState.GUESSER_PHASE)
    assert torch.equal(state.current_clue_index, clue_indices)
    assert torch.equal(state.current_clue_number, clue_numbers)
    assert torch.equal(state.remaining_guesses, clue_numbers + 1)


def test_guess_correct_color():
    """Test guessing a correct color."""
    batch_size = 1
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    # Give a clue first
    state.give_clue(torch.tensor([0], dtype=torch.int32), torch.tensor([2], dtype=torch.int32))

    # Find a tile with the current team's color
    team = state.current_team[0].item()
    team_tile_idx = torch.nonzero(state.colors[0] == team, as_tuple=False)[0, 0].item()

    initial_guesses = state.remaining_guesses[0].item()

    # Guess correct color
    state.guess(torch.tensor([team_tile_idx], dtype=torch.int32))

    # Check tile is revealed
    assert state.revealed[0, team_tile_idx]

    # Check guesses decremented
    assert state.remaining_guesses[0].item() == initial_guesses - 1

    # Still guesser phase (correct guess continues turn)
    assert state.phase[0].item() == GameState.GUESSER_PHASE


def test_guess_wrong_color_ends_turn():
    """Test that guessing wrong color ends the turn."""
    batch_size = 1
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    # Give a clue
    state.give_clue(torch.tensor([0], dtype=torch.int32), torch.tensor([2], dtype=torch.int32))

    # Find a tile with the opponent's color
    team = state.current_team[0].item()
    opponent = 1 - team
    opponent_tile_idx = torch.nonzero(state.colors[0] == opponent, as_tuple=False)[0, 0].item()

    initial_team = state.current_team[0].item()
    initial_turn = state.turn_count[0].item()

    # Guess opponent color
    state.guess(torch.tensor([opponent_tile_idx], dtype=torch.int32))

    # Check tile is revealed
    assert state.revealed[0, opponent_tile_idx]

    # Turn should have ended
    assert state.phase[0].item() == GameState.SPYMASTER_PHASE
    assert state.current_team[0].item() != initial_team
    assert state.turn_count[0].item() == initial_turn + 1
    assert state.remaining_guesses[0].item() == 0


def test_guess_assassin_ends_game():
    """Test that guessing assassin ends the game."""
    batch_size = 1
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    # Give a clue
    state.give_clue(torch.tensor([0], dtype=torch.int32), torch.tensor([2], dtype=torch.int32))

    # Find assassin tile
    assassin_idx = torch.nonzero(state.colors[0] == GameState.ASSASSIN, as_tuple=False)[0, 0].item()

    current_team = state.current_team[0].item()

    # Guess assassin
    state.guess(torch.tensor([assassin_idx], dtype=torch.int32))

    # Game should be over, opponent wins
    assert state.game_over[0]
    assert state.winner[0].item() == 1 - current_team


def test_victory_condition():
    """Test that finding all team words ends the game."""
    batch_size = 1
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    team = state.current_team[0].item()
    team_tiles = torch.nonzero(state.colors[0] == team, as_tuple=False).squeeze(-1)

    # Reveal all but one team tile manually
    for tile_idx in team_tiles[:-1]:
        state.revealed[0, tile_idx.item()] = True

    # Give clue and guess last team tile
    state.give_clue(torch.tensor([0], dtype=torch.int32), torch.tensor([1], dtype=torch.int32))
    state.guess(torch.tensor([team_tiles[-1].item()], dtype=torch.int32))

    # Game should be over, team wins
    assert state.game_over[0]
    assert state.winner[0].item() == team


def test_active_agent_masks():
    """Test that active agent masks are computed correctly."""
    batch_size = 4
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    # All games start in spymaster phase
    masks = state.get_active_agent_masks()

    # Check that correct agents are active
    for b in range(batch_size):
        team = state.current_team[b].item()
        if team == GameState.RED:
            assert masks["red_spy"][b]
            assert not masks["red_guess"][b]
            assert not masks["blue_spy"][b]
            assert not masks["blue_guess"][b]
        else:
            assert not masks["red_spy"][b]
            assert not masks["red_guess"][b]
            assert masks["blue_spy"][b]
            assert not masks["blue_guess"][b]

    # Give clues and check guesser masks
    state.give_clue(
        torch.arange(batch_size, dtype=torch.int32),
        torch.full((batch_size,), 2, dtype=torch.int32)
    )

    masks = state.get_active_agent_masks()

    for b in range(batch_size):
        team = state.current_team[b].item()
        if team == GameState.RED:
            assert not masks["red_spy"][b]
            assert masks["red_guess"][b]
            assert not masks["blue_spy"][b]
            assert not masks["blue_guess"][b]
        else:
            assert not masks["red_spy"][b]
            assert not masks["red_guess"][b]
            assert not masks["blue_spy"][b]
            assert masks["blue_guess"][b]


def test_end_turn_early():
    """Test ending turn early."""
    batch_size = 2
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    # Give clues
    state.give_clue(torch.tensor([0, 1], dtype=torch.int32), torch.tensor([2, 3], dtype=torch.int32))

    # End turn early for first game only
    mask = torch.tensor([True, False])
    initial_team = state.current_team.clone()

    state.end_turn_early(mask)

    # First game turn should have ended
    assert state.phase[0].item() == GameState.SPYMASTER_PHASE
    assert state.current_team[0].item() != initial_team[0].item()
    assert state.remaining_guesses[0].item() == 0

    # Second game should be unchanged
    assert state.phase[1].item() == GameState.GUESSER_PHASE
    assert state.current_team[1].item() == initial_team[1].item()
    assert state.remaining_guesses[1].item() == 4


def test_reset():
    """Test resetting the game state."""
    batch_size = 2
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    # Play some moves
    state.give_clue(torch.tensor([0, 1], dtype=torch.int32), torch.tensor([2, 3], dtype=torch.int32))
    team_tile = torch.nonzero(state.colors[0] == state.current_team[0], as_tuple=False)[0, 0].item()
    state.guess(torch.tensor([team_tile, 0], dtype=torch.int32))

    # Reset
    state.reset(seed=99)

    # Check everything is back to initial state
    assert not torch.any(state.revealed)
    assert not torch.any(state.game_over)
    assert torch.all(state.phase == GameState.SPYMASTER_PHASE)
    assert torch.all(state.remaining_guesses == 0)
    assert torch.all(state.turn_count == 0)


def test_unrevealed_counts():
    """Test getting unrevealed counts."""
    batch_size = 2
    state = GameState(batch_size=batch_size, seed=42, device='cpu')

    counts = state.get_unrevealed_counts()

    # Initially, all tiles unrevealed
    for b in range(batch_size):
        team = state.current_team[b].item()
        opponent = 1 - team

        if team == GameState.RED:
            assert counts["red"][b].item() == 9
            assert counts["blue"][b].item() == 8
        else:
            assert counts["red"][b].item() == 8
            assert counts["blue"][b].item() == 9

        assert counts["neutral"][b].item() == 7
        assert counts["assassin"][b].item() == 1

    # Reveal some tiles
    state.revealed[0, 0] = True
    state.revealed[0, 1] = True

    counts = state.get_unrevealed_counts()

    # Counts should decrease
    total_unrevealed_0 = sum(counts[c][0].item() for c in ["red", "blue", "neutral", "assassin"])
    assert total_unrevealed_0 == 23


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
