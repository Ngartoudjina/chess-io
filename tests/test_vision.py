import chess
import numpy as np
import pytest

from chessio.recognize import (
    SOURCE_COORDINATES,
    SOURCE_PAWNS,
    SOURCE_SETTING,
    SOURCE_TURN,
    Orientation,
    recognize,
)
from chessio.vision.board import BoardNotFoundError, crop_board, locate_board, split_cells
from chessio.vision.synthetic import PAGE_COLOR, random_position, render_board
from chessio.vision.templates import CalibrationError, TemplateSet, calibrate, white_at_bottom


def cells_of(image):
    return split_cells(crop_board(image, locate_board(image)))


@pytest.fixture(scope="module")
def templates():
    return calibrate(cells_of(render_board(chess.Board(), square=72)))


class TestLocateBoard:
    @pytest.mark.parametrize("square", [24, 50, 64, 90])
    def test_finds_board_at_any_scale(self, square):
        image = render_board(chess.Board(), square=square, margin=(37, 21, 140, 60))
        region = locate_board(image)
        assert abs(region.x - 37) <= 2
        assert abs(region.y - 21) <= 2
        assert abs(region.size - 8 * square) <= 3

    def test_board_filling_whole_capture(self):
        image = render_board(chess.Board(), square=60, margin=(0, 0, 0, 0))
        region = locate_board(image)
        assert (region.x, region.y) == (0, 0)
        assert abs(region.size - 480) <= 2

    def test_with_highlighted_squares(self):
        board = chess.Board()
        board.push_san("e4")
        image = render_board(board, highlight=(chess.E2, chess.E4))
        assert abs(locate_board(image).size - 512) <= 3

    def test_no_board(self):
        image = np.full((400, 600, 3), PAGE_COLOR, np.uint8)
        with pytest.raises(BoardNotFoundError):
            locate_board(image)

    def test_board_cut_by_capture_edge(self):
        image = render_board(chess.Board(), square=60, margin=(0, 0, 0, 0))
        with pytest.raises(BoardNotFoundError):
            locate_board(image[:, 150:])  # deux colonnes et demie coupées

    def test_too_small(self):
        image = render_board(chess.Board(), square=8, margin=(10, 10, 10, 10))
        with pytest.raises(BoardNotFoundError):
            locate_board(image)


class TestCalibration:
    def test_one_template_per_starting_piece(self, templates):
        assert len(templates.labels) == 32
        assert set(templates.labels) == set("PNBRQKpnbrqk")

    def test_detects_black_at_bottom(self):
        image = render_board(chess.Board(), white_bottom=False)
        assert not white_at_bottom(cells_of(image))
        flipped = calibrate(cells_of(image))
        assert set(flipped.labels) == set("PNBRQKpnbrqk")

    def test_rejects_non_initial_position(self):
        board = chess.Board()
        board.push_san("e4")
        with pytest.raises(CalibrationError):
            calibrate(cells_of(render_board(board)))

    def test_save_and_load(self, templates, tmp_path):
        path = tmp_path / "gabarits.npz"
        templates.save(path)
        loaded = TemplateSet.load(path)
        assert loaded.labels == templates.labels
        assert loaded.theme == templates.theme
        assert np.abs(loaded.images - templates.images).max() <= 1
        assert np.abs(loaded.masks - templates.masks).max() <= 1 / 255 + 1e-6


class TestRecognize:
    def test_start_position(self, templates):
        result = recognize(render_board(chess.Board()), "w", templates)
        assert result.board.fen() == chess.STARTING_FEN
        assert result.ok, result.issues

    @pytest.mark.parametrize("square", [40, 56, 64, 96])
    def test_random_positions_exact(self, templates, square):
        rng = np.random.default_rng(square)
        for _ in range(10):
            expected = random_position(rng)
            last = expected.peek() if expected.move_stack else None
            highlight = (last.from_square, last.to_square) if last else ()
            image = render_board(expected, square=square, highlight=highlight)
            result = recognize(
                image, expected.turn, templates, orientation=Orientation.WHITE_BOTTOM
            )
            assert result.board.board_fen() == expected.board_fen()
            assert not result.doubtful

    def test_black_at_bottom(self, templates):
        rng = np.random.default_rng(7)
        expected = random_position(rng, plies=31)  # trait aux Noirs
        image = render_board(expected, white_bottom=False)
        result = recognize(image, "b", templates)  # orientation déduite du trait
        assert not result.white_bottom
        assert result.board.board_fen() == expected.board_fen()

    def test_wrong_orientation_is_not_silently_accepted(self, templates):
        board = chess.Board()
        result = recognize(
            render_board(board), "w", templates, orientation=Orientation.BLACK_BOTTOM
        )
        # Plateau lu à l'envers : position différente, signalée ou du moins détectable.
        assert result.board.board_fen() != board.board_fen()

    def test_unknown_piece_is_doubtful(self, templates):
        image = render_board(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
        region = locate_board(image)
        s = region.size // 8
        # Tache rouge au centre de d4 (ligne 4, colonne 3 vue côté Blancs).
        y, x = region.y + 4 * s + s // 4, region.x + 3 * s + s // 4
        image[y : y + s // 2, x : x + s // 2] = (200, 30, 30)
        result = recognize(image, "w", templates)
        assert chess.D4 in result.doubtful
        assert not result.ok

    def test_recognition_under_500_ms(self, templates):
        image = render_board(random_position(np.random.default_rng(1)), square=90)
        recognize(image, "w", templates)  # préchauffage
        result = recognize(image, "w", templates)
        assert result.elapsed < 0.5


@pytest.fixture(scope="module")
def templates_without_coordinates():
    return calibrate(cells_of(render_board(chess.Board(), square=72, coordinates=False)))


class TestOrientation:
    """EF-04 : un plateau lu à l'envers donne une position légale mais absurde."""

    def test_coordinates_learned_at_calibration(self, templates, templates_without_coordinates):
        assert templates.glyphs is not None
        assert templates_without_coordinates.glyphs is None

    @pytest.mark.parametrize(("white_bottom", "turn"), [(True, "b"), (False, "w")])
    def test_coordinates_beat_side_to_move(self, templates, white_bottom, turn):
        # Le camp au trait n'est PAS en bas : seule la lecture des coordonnées sauve la mise.
        expected = random_position(np.random.default_rng(11), plies=40)
        image = render_board(expected, white_bottom=white_bottom)
        result = recognize(image, turn, templates)
        assert result.orientation_source == SOURCE_COORDINATES
        assert result.white_bottom == white_bottom
        assert result.board.board_fen() == expected.board_fen()

    # En deçà de ~36 px par case, les coordonnées (6 px de haut) deviennent illisibles :
    # l'orientation retombe alors sur la structure de pions, puis sur le trait.
    @pytest.mark.parametrize("square", [40, 64, 100])
    def test_coordinates_at_any_scale(self, templates, square):
        image = render_board(chess.Board(), square=square, white_bottom=False)
        result = recognize(image, "w", templates)
        assert (result.white_bottom, result.orientation_source) == (False, SOURCE_COORDINATES)

    def test_pawn_structure_without_coordinates(self, templates_without_coordinates):
        board = chess.Board()
        for san in "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7".split():
            board.push_san(san)
        image = render_board(board, white_bottom=False, coordinates=False)
        result = recognize(image, "w", templates_without_coordinates)
        assert (result.white_bottom, result.orientation_source) == (False, SOURCE_PAWNS)
        assert result.board.board_fen() == board.board_fen()

    def test_side_to_move_as_last_resort(self, templates_without_coordinates):
        board = chess.Board("8/8/4k3/8/8/2B2N2/8/4K3 b - - 0 1")  # aucun pion
        image = render_board(board, white_bottom=False, coordinates=False)
        result = recognize(image, "b", templates_without_coordinates)
        assert result.orientation_source == SOURCE_TURN
        assert result.orientation_assumed
        assert result.board.board_fen() == board.board_fen()

    def test_explicit_setting_wins(self, templates):
        result = recognize(
            render_board(chess.Board()), "w", templates, orientation=Orientation.WHITE_BOTTOM
        )
        assert result.orientation_source == SOURCE_SETTING
