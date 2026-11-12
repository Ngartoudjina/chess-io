"""Construction et validation de la FEN (EF-06, EF-07).

C'est le point de jonction entre la partie vision et la partie moteur :
la vision produit un placement (8 × 8 cases), ce module en fait une FEN
complète et vérifie qu'elle est jouable avant tout calcul.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import chess

# Cases initiales (roi, tour) et symbole FEN de chaque droit de roque.
_CASTLING_SQUARES: tuple[tuple[str, chess.Color, chess.Square, chess.Square], ...] = (
    ("K", chess.WHITE, chess.E1, chess.H1),
    ("Q", chess.WHITE, chess.E1, chess.A1),
    ("k", chess.BLACK, chess.E8, chess.H8),
    ("q", chess.BLACK, chess.E8, chess.A8),
)


@dataclass(frozen=True)
class Issue:
    """Un problème de validité, avec les cases à signaler à l'utilisateur."""

    message: str
    squares: tuple[chess.Square, ...] = field(default=())

    def __str__(self) -> str:
        if not self.squares:
            return self.message
        names = ", ".join(chess.square_name(s) for s in self.squares)
        return f"{self.message} ({names})"


class InvalidPositionError(ValueError):
    """La position ne peut pas être analysée ; ``issues`` détaille pourquoi."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        super().__init__("Position invalide : " + " ; ".join(str(i) for i in issues))


def parse_turn(turn: str | chess.Color) -> chess.Color:
    """Accepte ``w``/``b``, ``blancs``/``noirs`` ou directement ``chess.WHITE``/``BLACK``."""
    if isinstance(turn, bool):
        return turn
    key = turn.strip().lower()
    if key in {"w", "white", "blanc", "blancs"}:
        return chess.WHITE
    if key in {"b", "black", "noir", "noirs"}:
        return chess.BLACK
    raise ValueError(f"Couleur au trait inconnue : {turn!r} (attendu : w ou b)")


def infer_castling(board: chess.Board) -> str:
    """Déduit les droits de roque des cases initiales du roi et des tours.

    Ce n'est qu'une présomption : l'image ne dit pas si le roi ou la tour
    ont déjà bougé, d'où des champs modifiables avant le calcul (EF-06).
    """
    rights = ""
    for symbol, color, king_sq, rook_sq in _CASTLING_SQUARES:
        king = board.piece_at(king_sq)
        rook = board.piece_at(rook_sq)
        if king == chess.Piece(chess.KING, color) and rook == chess.Piece(chess.ROOK, color):
            rights += symbol
    return rights or "-"


def build_board(
    placement: str,
    turn: str | chess.Color,
    castling: str | None = None,
    en_passant: str = "-",
) -> chess.Board:
    """Construit l'échiquier à partir du placement et de la couleur au trait.

    ``castling=None`` déduit les droits de roque (voir :func:`infer_castling`).
    La prise en passant vaut « - » par défaut, invisible sur une capture.
    """
    board = chess.Board(None)
    board.set_board_fen(placement.strip())
    board.turn = parse_turn(turn)
    board.set_castling_fen(infer_castling(board) if castling is None else castling)
    board.ep_square = None if en_passant in {"", "-"} else chess.parse_square(en_passant)
    return board


def build_fen(
    placement: str,
    turn: str | chess.Color,
    castling: str | None = None,
    en_passant: str = "-",
) -> str:
    """Comme :func:`build_board`, mais renvoie directement la FEN."""
    return build_board(placement, turn, castling, en_passant).fen()


def _kings(board: chess.Board, color: chess.Color | None = None) -> tuple[chess.Square, ...]:
    colors = (chess.WHITE, chess.BLACK) if color is None else (color,)
    return tuple(sq for c in colors for sq in board.pieces(chess.KING, c))


def _pieces(board: chess.Board, color: chess.Color, piece_type: int | None = None):
    mask = board.occupied_co[color]
    if piece_type is not None:
        mask &= board.pieces_mask(piece_type, color)
    return tuple(chess.SquareSet(mask))


def _backrank_pawns(board: chess.Board) -> tuple[chess.Square, ...]:
    return tuple(chess.SquareSet(board.pawns & chess.BB_BACKRANKS))


def _checkers_of_waiting_king(board: chess.Board) -> tuple[chess.Square, ...]:
    king = board.king(not board.turn)
    if king is None:
        return ()
    return (king, *board.attackers(board.turn, king))


def _checkers_of_king_to_move(board: chess.Board) -> tuple[chess.Square, ...]:
    king = board.king(board.turn)
    return () if king is None else (king, *board.checkers())


def _castling_squares(board: chess.Board) -> tuple[chess.Square, ...]:
    squares: set[chess.Square] = set()
    for _, _, king_sq, rook_sq in _CASTLING_SQUARES:
        if board.castling_rights & chess.BB_SQUARES[rook_sq]:
            squares |= {king_sq, rook_sq}
    return tuple(sorted(squares))


def _ep_square(board: chess.Board) -> tuple[chess.Square, ...]:
    return () if board.ep_square is None else (board.ep_square,)


_Finder = Callable[[chess.Board], tuple[chess.Square, ...]]

# Chaque drapeau de python-chess → message en français et cases à signaler.
_STATUS_MESSAGES: tuple[tuple[chess.Status, str, _Finder], ...] = (
    (chess.STATUS_EMPTY, "Échiquier vide", lambda b: ()),
    (chess.STATUS_NO_WHITE_KING, "Aucun roi blanc", lambda b: ()),
    (chess.STATUS_NO_BLACK_KING, "Aucun roi noir", lambda b: ()),
    (chess.STATUS_TOO_MANY_KINGS, "Plus d'un roi dans un camp", _kings),
    (
        chess.STATUS_TOO_MANY_WHITE_PAWNS,
        "Plus de 8 pions blancs",
        lambda b: _pieces(b, chess.WHITE, chess.PAWN),
    ),
    (
        chess.STATUS_TOO_MANY_BLACK_PAWNS,
        "Plus de 8 pions noirs",
        lambda b: _pieces(b, chess.BLACK, chess.PAWN),
    ),
    (chess.STATUS_PAWNS_ON_BACKRANK, "Pion sur la 1re ou la 8e rangée", _backrank_pawns),
    (
        chess.STATUS_TOO_MANY_WHITE_PIECES,
        "Plus de 16 pièces blanches",
        lambda b: _pieces(b, chess.WHITE),
    ),
    (
        chess.STATUS_TOO_MANY_BLACK_PIECES,
        "Plus de 16 pièces noires",
        lambda b: _pieces(b, chess.BLACK),
    ),
    (
        chess.STATUS_OPPOSITE_CHECK,
        "Le camp qui n'a pas le trait est en échec (couleur au trait inversée ?)",
        _checkers_of_waiting_king,
    ),
    (
        chess.STATUS_TOO_MANY_CHECKERS,
        "Échec par plus de deux pièces, impossible",
        _checkers_of_king_to_move,
    ),
    (
        chess.STATUS_IMPOSSIBLE_CHECK,
        "Configuration d'échec impossible",
        _checkers_of_king_to_move,
    ),
    (
        chess.STATUS_BAD_CASTLING_RIGHTS,
        "Droits de roque incompatibles avec la position du roi ou des tours",
        _castling_squares,
    ),
    (chess.STATUS_INVALID_EP_SQUARE, "Case de prise en passant invalide", _ep_square),
)


def validate(board: chess.Board) -> list[Issue]:
    """Liste les problèmes qui empêchent d'analyser la position (vide si elle est valide)."""
    status = board.status()
    return [
        Issue(message, finder(board)) for flag, message, finder in _STATUS_MESSAGES if status & flag
    ]


def ensure_valid(board: chess.Board) -> None:
    """Lève :class:`InvalidPositionError` si la position n'est pas analysable."""
    issues = validate(board)
    if issues:
        raise InvalidPositionError(issues)
