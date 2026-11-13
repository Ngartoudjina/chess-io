"""Opérations d'image communes aux vignettes de cases (NumPy seul)."""

from __future__ import annotations

import numpy as np

RING = 3  # largeur du pourtour servant à estimer la couleur de fond, en pixels
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def background(cells: np.ndarray) -> np.ndarray:
    """Couleur de fond de chaque vignette : médiane de son pourtour (``... × 3``)."""
    lead = cells.shape[:-3]
    ring = np.concatenate(
        [
            cells[..., :RING, :, :].reshape(*lead, -1, 3),
            cells[..., -RING:, :, :].reshape(*lead, -1, 3),
            cells[..., RING:-RING, :RING, :].reshape(*lead, -1, 3),
            cells[..., RING:-RING, -RING:, :].reshape(*lead, -1, 3),
        ],
        axis=-2,
    )
    return np.median(ring, axis=-2).astype(np.float64)


def components(mask: np.ndarray, diagonal: bool) -> np.ndarray:
    """Étiquettes des composantes connexes (0 = hors masque). Pour de petites vignettes."""
    height, width = mask.shape
    inside = mask.tolist()
    labels = [[0] * width for _ in range(height)]
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    current = 0
    for y in range(height):
        for x in range(width):
            if not inside[y][x] or labels[y][x]:
                continue
            current += 1
            labels[y][x] = current
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                for dy, dx in steps:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and inside[ny][nx]
                        and not labels[ny][nx]
                    ):
                        labels[ny][nx] = current
                        stack.append((ny, nx))
    return np.array(labels, dtype=np.int32)


def blur(values: np.ndarray) -> np.ndarray:
    """Flou 3 × 3 (noyau 1-2-1) sur les deux derniers axes."""
    pad = [(0, 0)] * (values.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(values, pad, mode="edge")
    rows = (p[..., :-2, :] + 2 * p[..., 1:-1, :] + p[..., 2:, :]) / 4
    return (rows[..., :, :-2] + 2 * rows[..., :, 1:-1] + rows[..., :, 2:]) / 4


def reduce(values: np.ndarray) -> np.ndarray:
    """``(..., H, W)`` → ``(..., H/2, W/2)`` : moyenne 2 × 2 puis flou.

    Comparer à demi-résolution et légèrement flou rend la classification
    tolérante aux décalages d'une fraction de pixel que crée tout changement
    d'échelle (zoom du navigateur), sans effacer la forme des pièces.
    """
    *lead, h, w = values.shape
    return blur(values.reshape(*lead, h // 2, 2, w // 2, 2).mean(axis=(-3, -1)))


def reduce_rgb(values: np.ndarray) -> np.ndarray:
    """Comme :func:`reduce` pour des images ``(..., H, W, 3)``."""
    return np.moveaxis(reduce(np.moveaxis(values, -1, -3)), -3, -1)


def shift(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Décale une image de (dy, dx) pixels, en complétant par des zéros."""
    out = np.zeros_like(values)
    h, w = values.shape[:2]
    out[max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)] = values[
        max(0, -dy) : h + min(0, -dy), max(0, -dx) : w + min(0, -dx)
    ]
    return out
