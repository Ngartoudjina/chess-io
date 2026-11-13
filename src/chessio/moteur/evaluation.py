"""Évaluation statique : matériel et tables pièce-case, interpolées milieu / finale.

Point de départ sans entraînement (cahier des charges, section 8) : valeurs
inspirées de la « Simplified Evaluation Function » (Chessprogramming Wiki). Le
réglage Texel du lot L4 les ajustera sur des positions évaluées par Stockfish.

Le score est en centipions, du point de vue du camp au trait (convention négamax).
"""

from __future__ import annotations

import chess

# Valeur des pièces (centipions), identique en milieu de partie et en finale.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
BISHOP_PAIR = 30

# Tables vues côté Blancs, écrites comme on lit l'échiquier : 8e rangée en haut.
_PAWN_MG = (
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
)  # fmt: skip
_PAWN_EG = (
    0, 0, 0, 0, 0, 0, 0, 0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    15, 15, 15, 15, 15, 15, 15, 15,
    5, 5, 5, 5, 5, 5, 5, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
)  # fmt: skip
_KNIGHT = (
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
)  # fmt: skip
_BISHOP = (
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
)  # fmt: skip
_ROOK = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
)  # fmt: skip
_QUEEN = (
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
)  # fmt: skip
_KING_MG = (
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
)  # fmt: skip
_KING_EG = (
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
)  # fmt: skip

_TABLES = {
    chess.PAWN: (_PAWN_MG, _PAWN_EG),
    chess.KNIGHT: (_KNIGHT, _KNIGHT),
    chess.BISHOP: (_BISHOP, _BISHOP),
    chess.ROOK: (_ROOK, _ROOK),
    chess.QUEEN: (_QUEEN, _QUEEN),
    chess.KING: (_KING_MG, _KING_EG),
}

# Poids de chaque pièce dans la phase de jeu : 24 au départ, 0 en finale de rois et pions.
PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
MAX_PHASE = 24


def _square_values(table: tuple[int, ...], color: chess.Color, value: int) -> list[int]:
    """Matériel + table, indexés par case python-chess (a1 = 0) pour la couleur donnée."""
    values = [0] * 64
    for square in chess.SQUARES:
        file, rank = chess.square_file(square), chess.square_rank(square)
        row = 7 - rank if color == chess.WHITE else rank  # les Noirs lisent la table en miroir
        values[square] = value + table[row * 8 + file]
    return values


# MG[couleur][pièce][case] et EG[...] précalculés.
MG = {
    color: {pt: _square_values(_TABLES[pt][0], color, PIECE_VALUES[pt]) for pt in _TABLES}
    for color in chess.COLORS
}
EG = {
    color: {pt: _square_values(_TABLES[pt][1], color, PIECE_VALUES[pt]) for pt in _TABLES}
    for color in chess.COLORS
}


def phase(board: chess.Board) -> int:
    """Phase de jeu, de 24 (début) à 0 (finale sans pièces)."""
    total = (
        chess.popcount(board.knights | board.bishops)
        + 2 * chess.popcount(board.rooks)
        + 4 * chess.popcount(board.queens)
    )
    return min(total, MAX_PHASE)


def evaluate(board: chess.Board) -> int:
    """Score statique en centipions du point de vue du camp au trait."""
    mg = eg = 0
    for color in chess.COLORS:
        sign = 1 if color == chess.WHITE else -1
        occupied = board.occupied_co[color]
        mg_tables, eg_tables = MG[color], EG[color]
        for piece_type, mask in (
            (chess.PAWN, board.pawns),
            (chess.KNIGHT, board.knights),
            (chess.BISHOP, board.bishops),
            (chess.ROOK, board.rooks),
            (chess.QUEEN, board.queens),
            (chess.KING, board.kings),
        ):
            mg_values, eg_values = mg_tables[piece_type], eg_tables[piece_type]
            for square in chess.scan_reversed(mask & occupied):
                mg += sign * mg_values[square]
                eg += sign * eg_values[square]
        if chess.popcount(board.bishops & occupied) >= 2:
            mg += sign * BISHOP_PAIR
            eg += sign * BISHOP_PAIR
    game_phase = phase(board)
    # Troncature vers zéro (et non //) : le score d'une position et de son miroir
    # coloré restent exactement opposés.
    score = int((mg * game_phase + eg * (MAX_PHASE - game_phase)) / MAX_PHASE)
    return score if board.turn == chess.WHITE else -score
