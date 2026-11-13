"""Notation figurine : ♘f3 plutôt que Nf3, lisible quelle que soit la langue."""

from __future__ import annotations

from collections.abc import Sequence

import chess

_WHITE = str.maketrans("KQRBN", "♔♕♖♗♘")
_BLACK = str.maketrans("KQRBN", "♚♛♜♝♞")


def figurine(san: str, color: chess.Color) -> str:
    """Coup en notation algébrique (SAN) avec les pièces en figurines de la couleur du joueur."""
    return san.translate(_WHITE if color == chess.WHITE else _BLACK)


def figurine_line(board: chess.Board, moves: Sequence[chess.Move]) -> str:
    """Suite de coups numérotée : ``12. ♘f3 ♞c6 13. ♗b5`` (``12… ♞c6`` si les Noirs commencent)."""
    board = board.copy(stack=False)
    parts: list[str] = []
    for index, move in enumerate(moves):
        if board.turn == chess.WHITE:
            parts.append(f"{board.fullmove_number}.")
        elif index == 0:
            parts.append(f"{board.fullmove_number}…")
        parts.append(figurine(board.san(move), board.turn))
        board.push(move)
    return " ".join(parts)
