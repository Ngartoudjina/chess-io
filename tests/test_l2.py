"""Lot L2 : vision robuste, sans calibration, quel que soit le thème."""

import chess
import numpy as np
import pytest

from chessio.recognize import SOURCE_COORDINATES, TURN_LAST_MOVE, Orientation, recognize
from chessio.vision.board import BoardNotFoundError, detect_board
from chessio.vision.classify import load_model
from chessio.vision.coords import load_model as load_coordinates_model
from chessio.vision.synthetic import PAGE_COLOR, random_position, render_board
from chessio.vision.theme import Theme

# Thèmes absents des tests précédents, surlignage compris (couleurs Lichess, etc.).
THEMES = {
    "brun": Theme("brun", ((240, 217, 181), (205, 210, 106)), ((181, 136, 99), (170, 162, 58))),
    "bleu": Theme("bleu", ((222, 227, 230), (195, 216, 135)), ((140, 162, 173), (150, 170, 100))),
    "violet": Theme(
        "violet", ((239, 239, 239), (200, 220, 140)), ((136, 119, 183), (150, 150, 120))
    ),
}
needs_model = pytest.mark.skipif(load_model() is None, reason="modèle des cases non entraîné")


@pytest.mark.parametrize("name", THEMES)
def test_board_found_whatever_the_theme(name):
    image = render_board(chess.Board(), square=56, theme=THEMES[name], margin=(70, 40, 200, 90))
    region, theme = detect_board(image)
    assert (abs(region.x - 70), abs(region.y - 40), abs(region.size - 448)) <= (2, 2, 3)
    assert theme.light[0] != theme.dark[0]


def test_no_board_gives_a_clear_message():
    image = np.full((600, 400, 3), PAGE_COLOR, np.uint8)
    image[100:300, 50:350] = (60, 60, 200)  # un bouton, pas un échiquier
    with pytest.raises(BoardNotFoundError, match="Aucun échiquier"):
        detect_board(image)


@needs_model
@pytest.mark.parametrize("name", THEMES)
def test_reads_unseen_theme_without_calibration(name):
    rng = np.random.default_rng(len(name))
    for _ in range(3):
        board = random_position(rng)
        image = render_board(board, square=60, theme=THEMES[name])
        result = recognize(image, board.turn, None, orientation=Orientation.WHITE_BOTTOM)
        assert result.board.board_fen() == board.board_fen()


@needs_model
def test_last_move_in_any_highlight_color():
    board = chess.Board()
    for san in ("e4", "e5", "Nf3"):
        board.push_san(san)
    move = board.peek()
    image = render_board(board, theme=THEMES["brun"], highlight=(move.from_square, move.to_square))
    result = recognize(image, None, None)
    assert (result.board.turn, result.turn_source) == (chess.BLACK, TURN_LAST_MOVE)
    assert result.last_move == move


@pytest.mark.skipif(load_coordinates_model() is None, reason="modèle des coordonnées absent")
@needs_model
@pytest.mark.parametrize("white_bottom", [True, False])
def test_orientation_read_from_coordinates_without_calibration(white_bottom):
    board = random_position(np.random.default_rng(4), plies=30)
    image = render_board(board, square=72, white_bottom=white_bottom, theme=THEMES["bleu"])
    turn = chess.WHITE if not white_bottom else chess.BLACK  # le camp au trait n'est pas en bas
    result = recognize(image, turn, None)
    assert (result.white_bottom, result.orientation_source) == (white_bottom, SOURCE_COORDINATES)
    assert result.board.board_fen() == board.board_fen()
