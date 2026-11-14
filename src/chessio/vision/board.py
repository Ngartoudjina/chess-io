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

Les couleurs des cases ne sont pas imposées (lot L2) : :func:`detect_board` essaie
les thèmes connus, puis toutes les paires de couleurs dominantes de l'image ;
une paire n'est retenue que si elle dessine un vrai damier de 8 × 8 cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
from PIL import Image, ImageOps

from chessio.vision.theme import CHESSCOM_GREEN, Theme

CELL = 64  # côté d'une case normalisée, en pixels
MIN_SQUARE_PX = 12  # en dessous, la capture est trop petite pour être lue
COLOR_TOLERANCE = 20  # écart maximal par canal avec une couleur du thème
COARSE_FILL = 0.05  # part minimale de pixels du thème pour qu'une ligne compte
EDGE_FILL = 0.7  # part minimale de pixels du thème sur un bord (deux cases surlignées tolérées)
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
    """Charge une image (PNG, JPEG, WebP…) en tableau RGB ``uint8``, redressée selon
    son orientation EXIF (photos de téléphone)."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
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
    side = light.shape[0] // 8
    if side == 0:
        return 0

    def shares(mask: np.ndarray) -> np.ndarray:
        cells = mask[: 8 * side, : 8 * side].reshape(8, side, 8, side)
        return cells.mean(axis=(1, 3))  # part de chaque case, (8, 8)

    light_part, dark_part = shares(light), shares(dark)
    expected_light = (np.add.outer(np.arange(8), np.arange(8)) % 2) == 0
    good = (light_part + dark_part >= 0.3) & ((light_part > dark_part) == expected_light)
    return int(good.sum())


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


DOMINANT_COLORS = 10  # couleurs dominantes examinées pour trouver celles du damier
MIN_COLOR_SHARE = 0.004  # part minimale de l'image pour qu'une couleur compte
QUICK_SIDE = 360  # côté (pixels) de l'image réduite servant au tri des paires


def _dominant_colors(image: np.ndarray) -> list[tuple[int, int, int]]:
    """Couleurs moyennes des teintes les plus fréquentes (quantifiées sur 4 bits)."""
    pixels = image.reshape(-1, 3)
    keys = (
        (pixels[:, 0].astype(np.int32) >> 4 << 8) | (pixels[:, 1] >> 4 << 4) | (pixels[:, 2] >> 4)
    )
    counts = np.bincount(keys, minlength=4096)
    sums = [np.bincount(keys, weights=pixels[:, c], minlength=4096) for c in range(3)]
    colors = []
    for key in np.argsort(counts)[::-1][:DOMINANT_COLORS]:
        if counts[key] < MIN_COLOR_SHARE * len(keys):
            break
        colors.append(tuple(int(round(sums[c][key] / counts[key])) for c in range(3)))
    return colors


def _pair_score(
    mask_a: np.ndarray, mask_b: np.ndarray, first: tuple, second: tuple
) -> tuple[int, Theme] | None:
    """Sur l'image réduite : ces deux couleurs forment-elles un damier ? (score, thème)"""
    try:
        y0, y1, x0, x1 = _coarse_box(mask_a | mask_b)
    except BoardNotFoundError:
        return None
    height, width = y1 - y0 + 1, x1 - x0 + 1
    if min(height, width) < 8 * 3 or abs(height - width) > 0.12 * max(height, width):
        return None
    size = min(height, width)
    crop_a = mask_a[y0 : y0 + size, x0 : x0 + size]
    crop_b = mask_b[y0 : y0 + size, x0 : x0 + size]
    # La case en haut à gauche est claire : elle dit laquelle des deux couleurs l'est.
    corner = slice(0, max(1, size // 8))
    if crop_a[corner, corner].mean() >= crop_b[corner, corner].mean():
        light, dark, light_mask, dark_mask = first, second, crop_a, crop_b
    else:
        light, dark, light_mask, dark_mask = second, first, crop_b, crop_a
    consistent = _consistent_cells(light_mask, dark_mask)
    if consistent < MIN_CONSISTENT_CELLS - 4:
        return None
    theme = Theme(name="détecté", light=(light,), dark=(dark,))
    return consistent * size * size, theme


def candidate_themes(image: np.ndarray) -> list[Theme]:
    """Thèmes plausibles d'après les couleurs dominantes, du plus au moins probable."""
    step = max(1, min(image.shape[:2]) // QUICK_SIDE)
    small = image[::step, ::step]
    colors = _dominant_colors(small)
    index = color_index(small)
    masks = [color_mask(index, (color,)) for color in colors]
    scored = []
    for i, first in enumerate(colors):
        for j in range(i + 1, len(colors)):
            second = colors[j]
            if max(abs(a - b) for a, b in zip(first, second, strict=True)) <= 2 * COLOR_TOLERANCE:
                continue  # deux nuances d'une même couleur
            result = _pair_score(masks[i], masks[j], first, second)
            if result is not None:
                scored.append(result)
    scored.sort(key=lambda item: item[0], reverse=True)
    return [theme for _, theme in scored]


def detect_board(
    image: np.ndarray, preferred: tuple[Theme, ...] = (CHESSCOM_GREEN,)
) -> tuple[BoardRegion, Theme]:
    """Trouve le plateau quel que soit son thème : (zone, couleurs des cases).

    Les thèmes ``preferred`` (calibration, Chess.com) sont essayés d'abord, puis
    les paires de couleurs dominantes de l'image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image RGB attendue")
    for theme in preferred:
        try:
            return locate_board(image, theme), theme
        except BoardNotFoundError:
            pass
    for theme in candidate_themes(image)[:3]:
        try:
            return locate_board(image, theme), theme
        except BoardNotFoundError:
            pass
    raise BoardNotFoundError(
        "Aucun échiquier trouvé dans cette capture : il doit être entièrement visible, "
        "avec des cases de couleur unie."
    )


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
