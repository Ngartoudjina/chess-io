"""Parcours de l'interface, hors écran : calibration, trait déduit, correction d'une case."""

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import chess  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6.QtWidgets")

from chessio.engine import HOMEMADE, EngineNotFoundError, find_stockfish  # noqa: E402
from chessio.paths import templates_path  # noqa: E402
from chessio.vision.board import crop_board, locate_board, split_cells  # noqa: E402
from chessio.vision.synthetic import render_board  # noqa: E402
from chessio.vision.templates import calibrate  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from chessio.ui.app import create_app

    return create_app([])


@pytest.fixture
def make_window(app, tmp_path, monkeypatch):
    try:
        find_stockfish()
    except EngineNotFoundError:
        pytest.skip("Stockfish absent")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))  # calibration isolée du poste
    from PySide6.QtCore import QSettings

    # Réglages (moteur choisi, avertissement accepté) isolés eux aussi.
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path))
    from chessio.ui.window import MainWindow

    windows = []

    def factory(calibrated: bool):
        if calibrated:
            image = render_board(chess.Board())
            calibrate(split_cells(crop_board(image, locate_board(image)))).save(templates_path())
        window = MainWindow()
        window.resize(1200, 800)
        window.time_slider.setValue(5)  # 0,5 s par analyse
        window.show()
        windows.append(window)
        return window

    yield factory
    for window in windows:
        window.close()


def pump(app, condition, timeout=25.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not condition():
        app.processEvents()
        time.sleep(0.01)
    return condition()


def analysed(window):
    return lambda: window.analysis is not None and not window._progress_timer.isActive()


def test_first_capture_read_without_calibration(app, make_window):
    window = make_window(calibrated=False)
    assert window.templates is None  # réseau des cases : aucune calibration nécessaire
    board = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        board.push_san(san)
    last = board.peek()
    window._set_image(render_board(board, highlight=(last.from_square, last.to_square)))
    assert pump(app, analysed(window))
    assert window.board.board_fen() == board.board_fen()
    assert window.board.turn == chess.BLACK  # déduit du dernier coup surligné
    assert window.turn.value() == chess.BLACK
    assert window.analysis.best is not None


def test_doubtful_square_blocks_until_corrected(app, make_window):
    window = make_window(calibrated=True)
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    image = render_board(board)
    region = locate_board(image)
    size = region.size // 8
    y, x = region.y + 6 * size + size // 4, region.x + 4 * size + size // 4  # case e2
    image[y : y + size // 2, x : x + size // 2] = (200, 30, 30)
    window._set_image(image)
    assert pump(app, lambda: window.recognition is not None)
    assert chess.E2 in window.doubtful
    assert window.banner.isVisible() and window.banner.property("kind") == "warning"
    assert window.analysis is None  # aucun coup sur une position douteuse

    window._correct_square(chess.E2, "P")
    assert pump(app, analysed(window))
    assert not window.banner.isVisible()
    assert window.board.piece_at(chess.E2) == chess.Piece.from_symbol("P")


def test_fen_entry_is_analysed(app, make_window):
    window = make_window(calibrated=True)
    window.fen_edit.setText("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    window._load_fen()
    assert pump(app, analysed(window))
    assert window.analysis.best.uci == "a1a8"
    assert window.best_label.text() == "♖a8#"


def test_switch_to_homemade_engine(app, make_window):
    window = make_window(calibrated=True)
    assert pump(app, lambda: window.engine_ready)
    window.engine_choice.set_value(HOMEMADE)
    window._on_engine_changed(HOMEMADE)
    assert pump(app, lambda: window.engine_ready)
    assert "Chessio Moteur" in window.engine_chip.text()
    window.fen_edit.setText("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    window._load_fen()
    assert pump(app, analysed(window))
    assert window.analysis.best.uci == "a1a8"
