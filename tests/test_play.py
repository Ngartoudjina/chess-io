"""Jouer contre l'IA : coups de l'utilisateur, réponse de l'IA, indice, annulation."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import chess  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QPoint  # noqa: E402

from chessio.engine import EngineNotFoundError, find_stockfish  # noqa: E402


@pytest.fixture
def window(tmp_path):
    try:
        find_stockfish()
    except EngineNotFoundError:
        pytest.skip("Stockfish absent")
    from chessio.ui.app import create_app
    from chessio.ui.play import PlayWindow

    app = create_app([])
    w = PlayWindow()
    w.show()
    yield app, w
    w.close()


def pump(app, condition, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not condition():
        app.processEvents()
        time.sleep(0.01)
    return condition()


def click(w, uci):
    move = chess.Move.from_uci(uci)
    w._on_square(move.from_square, QPoint())
    w._on_square(move.to_square, QPoint())


def test_play_hint_and_undo(window):
    app, w = window
    assert pump(app, lambda: w.engines_ready)
    assert w.human == chess.WHITE
    click(w, "e2e4")
    assert pump(app, lambda: len(w.board.move_stack) == 2)  # l'IA a répondu
    assert w.board.turn == chess.WHITE
    assert pump(app, lambda: w.best_now is not None)  # analyse de l'assistant
    assert w.assistant.toPlainText()  # l'assistant a commenté la position
    w._ask_hint()
    assert pump(app, lambda: w.hint is not None)
    assert w.hint in w.board.legal_moves
    w._undo()
    assert w.board.move_stack == []


def test_illegal_click_is_ignored(window):
    app, w = window
    assert pump(app, lambda: w.engines_ready)
    click(w, "e2e5")  # coup illégal
    assert w.board.move_stack == []


def test_checkmate_ends_the_game(window, monkeypatch):
    app, w = window
    assert pump(app, lambda: w.engines_ready)
    monkeypatch.setattr("chessio.ui.play.QMessageBox.information", lambda *a: None)
    w.board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    w._after_move()
    click(w, "a1a8")
    assert w.board.is_checkmate()
    assert "gagnez" in w.status.text()
