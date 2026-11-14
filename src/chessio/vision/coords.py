"""Lecture des coordonnées écrites sur l'échiquier, sans calibration (lot L2, EF-04).

Les sites et applications écrivent le chiffre de la rangée en haut à gauche des
cases de la colonne de gauche, et la lettre de la colonne en bas à droite des
cases de la rangée du bas. Un petit réseau (même architecture que celui des
cases) reconnaît ce caractère — ou son absence — dans n'importe quelle police,
couleur ou contraste.

Chaque coordonnée lue vote pour « Blancs en bas » (8 en haut à gauche, « a » en
bas à gauche) ou « Noirs en bas » (1 et « h ») selon la probabilité du caractère
attendu dans chaque hypothèse ; les coins vides ou illisibles s'annulent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
from PIL import Image

from chessio.vision.board import CELL, screen_to_square
from chessio.vision.cnn import TileNet, features_from_reduced
from chessio.vision.ops import background

GLYPHS = (".", "1", "2", "3", "4", "5", "6", "7", "8", "a", "b", "c", "d", "e", "f", "g", "h")
MODEL_PATH = Path(__file__).with_name("modeles") / "coordonnees.npz"
CORNER = 24  # côté de la zone examinée dans une case de 64 pixels
DIGIT_ZONE = (slice(0, CORNER), slice(0, CORNER))  # en haut à gauche
LETTER_ZONE = (slice(CELL - CORNER, CELL), slice(CELL - CORNER, CELL))  # en bas à droite
MIN_MARGIN = 4.0  # écart de log-vraisemblance exigé entre les deux orientations
MIN_READABLE = 3  # coordonnées lues avec assurance nécessaires pour conclure


def corner_features(cells: np.ndarray, zone: tuple[slice, slice]) -> np.ndarray:
    """Coins des cases ``(k, 64, 64, 3)`` → entrées du réseau ``(k, 32, 32, 6)``."""
    cells = cells.reshape(-1, CELL, CELL, 3)
    bg = background(cells) / 255
    patches = np.stack(
        [
            np.asarray(
                Image.fromarray(np.ascontiguousarray(cell[zone])).resize(
                    (32, 32), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            for cell in cells
        ]
    )
    return features_from_reduced(patches / 255, bg)


@lru_cache(maxsize=1)
def load_model() -> TileNet | None:
    return TileNet.load(MODEL_PATH) if MODEL_PATH.exists() else None


def read_orientation(cells: np.ndarray, net: TileNet | None = None) -> bool | None:
    """Vrai si les Blancs sont en bas, faux si ce sont les Noirs, ``None`` si illisible.

    ``cells`` : les 8 × 8 cases en ordre de l'écran.
    """
    net = net if net is not None else load_model()
    if net is None:
        return None
    left = net.predict(corner_features(cells[:, 0], DIGIT_ZONE))  # (8, 17)
    bottom = net.predict(corner_features(cells[7, :], LETTER_ZONE))  # (8, 17)
    readable = int((left[:, 0] < 0.5).sum() + (bottom[:, 0] < 0.5).sum())
    if readable < MIN_READABLE:
        return None
    scores = {}
    for white_bottom in (True, False):
        total = 0.0
        for row in range(8):
            rank = chess.square_rank(screen_to_square(row, 0, white_bottom))
            total += np.log(left[row, GLYPHS.index(str(rank + 1))] + 1e-4)
        for col in range(8):
            file = chess.square_file(screen_to_square(7, col, white_bottom))
            total += np.log(bottom[col, GLYPHS.index(chess.FILE_NAMES[file])] + 1e-4)
        scores[white_bottom] = total
    margin = scores[True] - scores[False]
    return None if abs(margin) < MIN_MARGIN else bool(margin > 0)
