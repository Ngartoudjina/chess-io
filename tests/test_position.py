import chess
import pytest

from chessio.position import (
    InvalidPositionError,
    build_fen,
    ensure_valid,
    infer_castling,
    parse_turn,
    validate,
)

START_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def squares(*names: str) -> set[int]:
    return {chess.parse_square(n) for n in names}


def issue_squares(fen: str) -> set[int]:
    return {sq for issue in validate(chess.Board(fen)) for sq in issue.squares}


class TestBuildFen:
    def test_start_position(self):
        assert build_fen(START_PLACEMENT, "w") == chess.STARTING_FEN

    def test_black_to_move(self):
        fen = build_fen(START_PLACEMENT, "b")
        assert fen.split()[1] == "b"

    def test_en_passant_defaults_to_dash(self):
        fen = build_fen("4k3/8/8/3pP3/8/8/8/4K3", "w")
        assert fen.split()[3] == "-"

    def test_explicit_castling_overrides_inference(self):
        fen = build_fen(START_PLACEMENT, "w", castling="Kq")
        assert fen.split()[2] == "Kq"

    @pytest.mark.parametrize(
        ("placement", "expected"),
        [
            (START_PLACEMENT, "KQkq"),
            ("r3k2r/8/8/8/8/8/8/R3K2R", "KQkq"),
            ("r3k3/8/8/8/8/8/8/4K2R", "Kq"),  # tour h8 et tour a1 absentes
            ("r3k2r/8/8/8/8/8/8/R4K1R", "kq"),  # roi blanc déplacé
            ("4k3/8/8/8/8/8/8/4K3", "-"),
            ("r3k2R/8/8/8/8/8/8/4K3", "q"),  # tour blanche en h8 : pas de droit noir
        ],
    )
    def test_castling_inference(self, placement, expected):
        board = chess.Board(None)
        board.set_board_fen(placement)
        assert infer_castling(board) == expected

    @pytest.mark.parametrize("turn", ["w", "W", "blancs", "white", chess.WHITE])
    def test_parse_turn_white(self, turn):
        assert parse_turn(turn) is chess.WHITE

    @pytest.mark.parametrize("turn", ["b", "noirs", "black", chess.BLACK])
    def test_parse_turn_black(self, turn):
        assert parse_turn(turn) is chess.BLACK

    def test_parse_turn_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_turn("x")


class TestValidate:
    def test_start_position_is_valid(self):
        assert validate(chess.Board()) == []
        ensure_valid(chess.Board())

    def test_two_white_kings(self):
        fen = "4k3/8/8/8/8/8/8/K3K3 w - - 0 1"
        assert squares("a1", "e1") <= issue_squares(fen)

    def test_missing_black_king(self):
        issues = validate(chess.Board("8/8/8/8/8/8/8/4K3 w - - 0 1"))
        assert any("roi noir" in i.message for i in issues)

    def test_pawn_on_eighth_rank(self):
        fen = "3Pk3/8/8/8/8/8/8/4K3 w - - 0 1"
        assert squares("d8") <= issue_squares(fen)

    def test_pawn_on_first_rank(self):
        fen = "4k3/8/8/8/8/8/8/p3K3 w - - 0 1"
        assert squares("a1") <= issue_squares(fen)

    def test_side_not_to_move_in_check(self):
        # Les Noirs sont en échec alors que c'est aux Blancs de jouer :
        # typiquement une couleur au trait mal saisie.
        fen = "4k3/8/8/8/8/8/8/4RK2 w - - 0 1"
        issues = validate(chess.Board(fen))
        assert any("trait" in i.message for i in issues)
        assert squares("e8", "e1") <= issue_squares(fen)

    def test_same_position_valid_with_other_side_to_move(self):
        assert validate(chess.Board("4k3/8/8/8/8/8/8/4RK2 b - - 0 1")) == []

    def test_nine_white_pawns(self):
        fen = "4k3/8/8/8/8/P7/PPPPPPPP/4K3 w - - 0 1"
        assert any("pions blancs" in i.message for i in validate(chess.Board(fen)))

    def test_ensure_valid_raises_with_issues(self):
        with pytest.raises(InvalidPositionError) as exc_info:
            ensure_valid(chess.Board("4k3/8/8/8/8/8/8/K3K3 w - - 0 1"))
        assert exc_info.value.issues
        assert "e1" in str(exc_info.value)
