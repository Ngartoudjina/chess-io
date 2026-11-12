import pytest

from chessio.__main__ import main
from chessio.engine import EngineNotFoundError, find_stockfish


def test_invalid_position_returns_error_and_lists_squares(capsys):
    code = main(["analyse", "4k3/8/8/8/8/8/8/K3K3 w - - 0 1"])
    assert code == 2
    assert "e1" in capsys.readouterr().err


def test_placement_without_turn_is_rejected(capsys):
    code = main(["analyse", "4k3/8/8/8/8/8/8/4K3"])
    assert code == 2
    assert "--trait" in capsys.readouterr().err


def test_analyse_prints_best_move(capsys):
    try:
        find_stockfish()
    except EngineNotFoundError:
        pytest.skip("Stockfish absent")
    code = main(["analyse", "6k1/5ppp/8/8/8/8/8/R5K1", "--trait", "w", "--temps", "0.3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Ra8#" in out
    assert "(a1a8)" in out
