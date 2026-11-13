"""Localisation de l'échiquier dans une capture et découpe en 64 cases (EF-02).

Méthode, en trois temps :

1. **Repérage grossier** sur une image sous-échantillonnée : on marque les
   pixels aux couleurs des cases du thème, puis on cherche la plus longue
   plage de lignes et de colonnes qui en contiennent.
2. **Bords exacts** en pleine résolution : le pourtour extérieur du plateau
   n'est jamais recouvert par les pièces, donc les premières et dernières
   lignes (et colonnes) presque entièrement aux couleurs du thème sont les
   bords du plateau.
3. **Vérification** : la zone doit être carrée et ses 64 cases doivent
   alterner clair et foncé. La case en haut à gauche de l'écran est toujours
   claire (a8 vue côté Blancs, h1 vue côté Noirs).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
from PIL import Image

from chessio.vision.theme import CHESSCOM_GREEN, Theme

CELL = 64  # côté d'une case normalisée, en pixels
MIN_SQUARE_PX = 12  # en dessous, la capture est trop petite pour être lue
COLOR_TOLERANCE = 20  # écart maximal par canal avec une couleur du thème
COARSE_FILL = 0.05  # part minimale de pixels du thème pour qu'une ligne compte
EDGE_FILL = 0.8  # part minimale de pixels du thème sur un bord du plateau
MIN_CONSISTENT_CELLS = 60  # cases (sur 64) qui doivent respecter l'alternance


class BoardNotFoundError(ValueError):
    """Aucun échiquier complet n'a été trouvé dans la capture."""


@dataclass(frozen=True)
class BoardRegion:
    """Position du plateau dans la capture, en pixels (carré de côté ``size``)."""

    x: int
    y: int
    size: int

    @property
    def square(self) -> float:
        return self.size / 8


def load_image(path: str | Path) -> np.ndarray:
    """Charge un PNG ou un JPEG en tableau RGB ``uint8``."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


@lru_cache(maxsize=32)
def _color_table(colors: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    """Table « couleur quantifiée (6 bits par canal) → proche d'une des couleurs »."""
    table = np.zeros((64, 64, 64), dtype=bool)
    for color in colors:
        lo = [max(0, v - COLOR_TOLERANCE) >> 2 for v in color]
        hi = [min(255, v + COLOR_TOLERANCE) >> 2 for v in color]
        table[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = True
    return table.ravel()


def color_index(image: np.ndarray) -> np.ndarray:
    """Indice de chaque pixel dans les tables de couleurs (calculé une fois par image)."""
    q = (image >> 2).astype(np.int32)
    return (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]


def color_mask(index: np.ndarray, colors: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    """Pixels proches (à ``COLOR_TOLERANCE`` près, par canal) d'une des couleurs."""
    return _color_table(tuple(colors))[index]


def _longest_run(profile: np.ndarray, gap: int) -> tuple[int, int] | None:
    """Plus longue plage d'indices où ``profile`` est notable, trous < ``gap`` comblés."""
    if profile.max() <= 0:
        return None
    idx = np.flatnonzero(profile >= COARSE_FILL * profile.max())
    breaks = np.flatnonzero(np.diff(idx) > gap)
    starts = np.concatenate([idx[:1], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], idx[-1:]])
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def _coarse_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) inclusifs de la plus grande zone aux couleurs du thème."""
    height, width = mask.shape
    y0, y1, x0, x1 = 0, height - 1, 0, width - 1
    for _ in range(2):
        rows = _longest_run(mask[:, x0 : x1 + 1].sum(axis=1), gap=max(2, height // 30))
        if rows is None:
            break
        y0, y1 = rows
        cols = _longest_run(mask[y0 : y1 + 1].sum(axis=0), gap=max(2, width // 30))
        if cols is None:
            break
        x0, x1 = cols
    else:
        return y0, y1, x0, x1
    raise BoardNotFoundError(
        "Aucun échiquier reconnu : thème différent, capture trop petite ou plateau masqué."
    )


def _full_lines(mask: np.ndarray, axis: int) -> tuple[int, int] | None:
    """Premier et dernier indice de ligne (axis=1) ou colonne (axis=0) presque pleine."""
    full = np.flatnonzero(mask.mean(axis=axis) >= EDGE_FILL)
    return (int(full[0]), int(full[-1])) if full.size else None


def _consistent_cells(light: np.ndarray, dark: np.ndarray) -> int:
    """Nombre de cases dont la couleur dominante respecte l'alternance de l'échiquier."""
    edges = np.linspace(0, light.shape[0], 9).round().astype(int)
    count = 0
    for row in range(8):
        for col in range(8):
            ys = slice(edges[row], edges[row + 1])
            xs = slice(edges[col], edges[col + 1])
            light_part, dark_part = light[ys, xs].mean(), dark[ys, xs].mean()
            expected_light = (row + col) % 2 == 0
            if light_part + dark_part >= 0.3 and (light_part > dark_part) == expected_light:
                count += 1
    return count


def locate_board(image: np.ndarray, theme: Theme = CHESSCOM_GREEN) -> BoardRegion:
    """Trouve le plateau dans une capture RGB (``H × W × 3``, ``uint8``)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image RGB attendue")
    height, width = image.shape[:2]

    # 1. Repérage grossier (un pixel sur ``step`` suffit, les couleurs restent exactes).
    step = max(1, min(height, width) // 400)
    small = color_index(image[::step, ::step])
    y0, y1, x0, x1 = _coarse_box(color_mask(small, theme.light) | color_mask(small, theme.dark))
    top, bottom = max(0, (y0 - 1) * step), min(height, (y1 + 2) * step)
    left, right = max(0, (x0 - 1) * step), min(width, (x1 + 2) * step)

    # 2. Bords exacts en pleine résolution.
    window = color_index(image[top:bottom, left:right])
    light, dark = color_mask(window, theme.light), color_mask(window, theme.dark)
    both = light | dark
    rows = _full_lines(both, axis=1)
    cols = _full_lines(both[rows[0] : rows[1] + 1], axis=0) if rows else None
    rows = _full_lines(both[:, cols[0] : cols[1] + 1], axis=1) if cols else None
    if rows is None or cols is None:
        raise BoardNotFoundError("Bords du plateau introuvables.")
    (r0, r1), (c0, c1) = rows, cols
    board_h, board_w = r1 - r0 + 1, c1 - c0 + 1

    # 3. Vérifications : carré, taille suffisante, alternance des cases.
    if abs(board_w - board_h) > max(3, 0.02 * max(board_w, board_h)):
        touches_edge = (
            top + r0 == 0 or left + c0 == 0 or top + r1 == height - 1 or left + c1 == width - 1
        )
        raise BoardNotFoundError(
            "Plateau coupé par le bord de la capture : il doit être entièrement visible."
            if touches_edge
            else "La zone détectée n'est pas carrée : est-ce bien un échiquier ?"
        )
    size = min(board_w, board_h)
    if size < 8 * MIN_SQUARE_PX:
        raise BoardNotFoundError("Plateau trop petit sur la capture : agrandissez-le.")
    region_light = light[r0 : r0 + size, c0 : c0 + size]
    region_dark = dark[r0 : r0 + size, c0 : c0 + size]
    if _consistent_cells(region_light, region_dark) < MIN_CONSISTENT_CELLS:
        raise BoardNotFoundError("La zone détectée n'a pas l'alternance des cases d'un échiquier.")
    return BoardRegion(left + c0, top + r0, size)


def crop_board(image: np.ndarray, region: BoardRegion) -> np.ndarray:
    """Plateau recadré et remis à l'échelle (``8·CELL`` pixels de côté)."""
    crop = image[region.y : region.y + region.size, region.x : region.x + region.size]
    if min(crop.shape[:2]) < 8 * MIN_SQUARE_PX:
        raise BoardNotFoundError("Plateau trop petit sur la capture.")
    target = 8 * CELL
    # Réduction : moyenne par zone (BOX) ; agrandissement : bilinéaire.
    resample = Image.Resampling.BOX if crop.shape[0] > target else Image.Resampling.BILINEAR
    resized = Image.fromarray(np.ascontiguousarray(crop)).resize((target, target), resample)
    return np.asarray(resized, dtype=np.uint8)


def split_cells(board: np.ndarray) -> np.ndarray:
    """Découpe un plateau normalisé en ``8 × 8`` vignettes (ordre de l'écran)."""
    return board.reshape(8, CELL, 8, CELL, 3).swapaxes(1, 2)


def screen_to_square(row: int, col: int, white_bottom: bool) -> chess.Square:
    """Case de l'échiquier affichée à la ligne ``row``, colonne ``col`` de l'écran."""
    if white_bottom:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)
