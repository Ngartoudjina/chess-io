"""Dernier coup joué, d'après le surlignage de Chess.com.

Chess.com surligne en jaune la case de départ et la case d'arrivée du dernier
coup. Après le coup, la case de départ est vide et la case d'arrivée porte la
pièce qui vient de jouer : sa couleur dit qui a joué, donc qui a le trait. Si
c'était un pion avançant de deux cases, la prise en passant devient possible.
"""

from __future__ import annotations

import numpy as np

from chessio.vision.ops import background
from chessio.vision.theme import Theme

HIGHLIGHT_TOLERANCE = 25.0  # écart maximal entre le fond d'une case et le surlignage
EMPTY = "."

Cell = tuple[int, int]  # (ligne, colonne) à l'écran


def highlighted_cells(cells: np.ndarray, theme: Theme) -> list[Cell]:
    """Cases dont le fond a la couleur du surlignage du dernier coup."""
    bg = background(cells.astype(np.float64))  # (8, 8, 3)
    marks = np.asarray(theme.highlight, np.float64)  # (h, 3)
    distance = np.linalg.norm(bg[:, :, None, :] - marks[None, None], axis=-1).min(axis=-1)
    return [
        (row, col)
        for row in range(8)
        for col in range(8)
        if distance[row, col] <= HIGHLIGHT_TOLERANCE
    ]


def last_move(cells: np.ndarray, labels: list[list[str]], theme: Theme) -> tuple[Cell, Cell] | None:
    """(départ, arrivée) du dernier coup, ou ``None`` si le surlignage est absent ou ambigu.

    Il faut exactement deux cases surlignées, l'une vide (départ) et l'autre
    occupée (arrivée) ; trois cases (une pièce sélectionnée en plus) ou deux
    cases occupées ne permettent pas de conclure.
    """
    marked = highlighted_cells(cells, theme)
    if len(marked) != 2:
        return None
    first, second = marked
    first_empty = labels[first[0]][first[1]] == EMPTY
    second_empty = labels[second[0]][second[1]] == EMPTY
    if first_empty == second_empty:
        return None
    return (first, second) if first_empty else (second, first)
