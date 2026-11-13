"""Perft : nombre de positions atteignables à une profondeur donnée.

Vérifie la génération des coups contre les valeurs de référence (cahier des
charges, section 9 : 8 902 nœuds à la profondeur 3 et 4 865 609 à la profondeur 5
depuis la position initiale).
"""

from __future__ import annotations

import chess


def perft(board: chess.Board, depth: int) -> int:
    """Nombre de feuilles à ``depth`` demi-coups (comptage direct au dernier niveau)."""
    if depth <= 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        try:
            total += perft(board, depth - 1)
        finally:
            board.pop()
    return total


def divide(board: chess.Board, depth: int) -> dict[str, int]:
    """Perft détaillé par premier coup (pour localiser une divergence avec une référence)."""
    counts: dict[str, int] = {}
    for move in board.legal_moves:
        board.push(move)
        try:
            counts[move.uci()] = perft(board, depth - 1)
        finally:
            board.pop()
    return counts
