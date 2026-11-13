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


# --- structure de pions, tours, roi, mobilité (v0.2) ---------------------------

# Bonus (milieu, finale) d'un pion passé selon le nombre de rangées déjà parcourues.
PASSED_BONUS = ((0, 0), (5, 10), (10, 15), (15, 25), (25, 45), (40, 75), (60, 120), (0, 0))
DOUBLED_PAWN = (-10, -20)
ISOLATED_PAWN = (-12, -15)
ROOK_OPEN_FILE = (25, 10)
ROOK_SEMI_OPEN_FILE = (12, 6)
KING_SHIELD_PAWN = 12  # milieu de partie seulement
# Mobilité : (milieu, finale) par case accessible, au-delà d'une mobilité « normale ».
MOBILITY = {
    chess.KNIGHT: (4, 4, 4),
    chess.BISHOP: (5, 5, 7),
    chess.ROOK: (2, 4, 7),
    chess.QUEEN: (1, 2, 14),
}

FILE_MASKS = [chess.BB_FILES[file] for file in range(8)]
ADJACENT_FILES = [
    (FILE_MASKS[file - 1] if file > 0 else 0) | (FILE_MASKS[file + 1] if file < 7 else 0)
    for file in range(8)
]


def _passed_masks(color: chess.Color) -> list[int]:
    """Cases devant un pion (sa colonne et les voisines) où un pion adverse l'arrêterait."""
    masks = []
    for square in chess.SQUARES:
        file, rank = chess.square_file(square), chess.square_rank(square)
        ahead = range(rank + 1, 8) if color == chess.WHITE else range(0, rank)
        files = FILE_MASKS[file] | ADJACENT_FILES[file]
        mask = 0
        for r in ahead:
            mask |= chess.BB_RANKS[r] & files
        masks.append(mask)
    return masks


PASSED_MASKS = {color: _passed_masks(color) for color in chess.COLORS}


def _shield_masks(color: chess.Color) -> list[int]:
    """Les trois cases devant le roi (sa colonne et les voisines), sur la rangée suivante."""
    masks = []
    for square in chess.SQUARES:
        file, rank = chess.square_file(square), chess.square_rank(square)
        front = rank + 1 if color == chess.WHITE else rank - 1
        masks.append(
            chess.BB_RANKS[front] & (FILE_MASKS[file] | ADJACENT_FILES[file])
            if 0 <= front < 8
            else 0
        )
    return masks


SHIELD_MASKS = {color: _shield_masks(color) for color in chess.COLORS}


def _structure(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    """Pions (passés, doublés, isolés), tours sur colonne ouverte, abri du roi, mobilité."""
    mg = eg = 0
    own = board.occupied_co[color]
    own_pawns = board.pawns & own
    their_pawns = board.pawns & board.occupied_co[not color]
    passed = PASSED_MASKS[color]
    for square in chess.scan_reversed(own_pawns):
        file = chess.square_file(square)
        if not their_pawns & passed[square]:
            advanced = (
                chess.square_rank(square) if color == chess.WHITE else 7 - chess.square_rank(square)
            )
            bonus = PASSED_BONUS[advanced]
            mg += bonus[0]
            eg += bonus[1]
        if not own_pawns & ADJACENT_FILES[file]:
            mg += ISOLATED_PAWN[0]
            eg += ISOLATED_PAWN[1]
    for file in range(8):
        count = chess.popcount(own_pawns & FILE_MASKS[file])
        if count > 1:
            mg += DOUBLED_PAWN[0] * (count - 1)
            eg += DOUBLED_PAWN[1] * (count - 1)
    for square in chess.scan_reversed(board.rooks & own):
        file_mask = FILE_MASKS[chess.square_file(square)]
        if not own_pawns & file_mask:
            bonus = ROOK_OPEN_FILE if not their_pawns & file_mask else ROOK_SEMI_OPEN_FILE
            mg += bonus[0]
            eg += bonus[1]
    king = board.king(color)
    if king is not None:
        mg += KING_SHIELD_PAWN * chess.popcount(own_pawns & SHIELD_MASKS[color][king])
    # Mobilité : cases attaquées non occupées par ses propres pièces.
    for piece_type, (mg_weight, eg_weight, typical) in MOBILITY.items():
        for square in chess.scan_reversed(board.pieces_mask(piece_type, color)):
            moves = chess.popcount(board.attacks_mask(square) & ~own) - typical
            mg += mg_weight * moves
            eg += eg_weight * moves
    return mg, eg


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
        structure_mg, structure_eg = _structure(board, color)
        mg += sign * structure_mg
        eg += sign * structure_eg
    game_phase = phase(board)
    # Troncature vers zéro (et non //) : le score d'une position et de son miroir
    # coloré restent exactement opposés.
    score = int((mg * game_phase + eg * (MAX_PHASE - game_phase)) / MAX_PHASE)
    return score if board.turn == chess.WHITE else -score
