"""Classification des cases par correspondance de gabarits (lot L1, EF-05).

Les gabarits sont extraits d'une seule capture de la position initiale
(« calibration ») : chaque pièce y apparaît au moins une fois. Pour comparer
une case à un gabarit, on recompose le gabarit sur le fond *de la case
examinée* (estimé sur son pourtour) : un roi noir calibré sur case claire
reconnaît donc aussi un roi noir sur case foncée ou surlignée.

Le classifieur ONNX du lot L2 remplacera ce module sans changer son interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import chess
import numpy as np

from chessio.vision.board import CELL, screen_to_square
from chessio.vision.ops import NEIGHBOURS, background, components, reduce, reduce_rgb, shift
from chessio.vision.orientation import CoordinateGlyphs, learn_glyphs
from chessio.vision.theme import CHESSCOM_GREEN, Theme

EMPTY = "."
PIECE_THRESHOLD = 22.0  # écart au fond au-delà duquel un pixel appartient à la pièce
MATTE_REACH = 4  # distance (pixels) jusqu'où l'on cherche la couleur de la pièce
MIN_CONTRAST = 25.0  # en deçà, pièce et fond trop proches pour estimer l'opacité

# Une case est douteuse si rien ne lui ressemble vraiment (erreur absolue)
# ou si deux classes sont presque aussi plausibles (marge relative).
# Réglage : sur captures synthétiques (cases de 26 à 96 px, JPEG jusqu'à q40),
# aucune erreur et des cases justes au-dessus de 0,20 de confiance ; sur les
# 20 captures réelles de la recette L1, aucune case juste sous 0,25.
MAX_RMS = 28.0
MIN_CONFIDENCE = 0.25


def _comparison_weights() -> np.ndarray:
    """Poids des pixels dans la comparaison : 0 dans les quatre coins de la case.

    Chess.com y dessine des éléments étrangers à la pièce : les coordonnées
    (chiffre en haut à gauche, lettre en bas à droite) et, en fin de partie,
    des pastilles sur les rois (mat, victoire) qui débordent sur les cases
    voisines. Les pièces, centrées, n'atteignent presque pas les coins.
    """
    weights = np.ones((CELL, CELL), np.float64)
    top, bottom = int(0.33 * CELL), int(0.74 * CELL)
    left, right = int(0.25 * CELL), int(0.75 * CELL)
    weights[:top, :left] = 0  # chiffre de la rangée
    weights[:top, right:] = 0  # pastille d'un roi
    weights[bottom:, :left] = 0  # pastille débordant d'une case voisine
    weights[bottom:, right:] = 0  # lettre de la colonne
    return weights


WEIGHTS = _comparison_weights()


class CalibrationError(ValueError):
    """La capture fournie pour la calibration n'est pas exploitable."""


@dataclass(frozen=True)
class CellPrediction:
    label: str  # symbole FEN de la pièce, ou « . » pour une case vide
    confidence: float  # 0 (ambigu) à 1 (sans hésitation)
    rms: float  # écart moyen au meilleur gabarit, en niveaux de couleur

    @property
    def doubtful(self) -> bool:
        return self.confidence < MIN_CONFIDENCE or self.rms > MAX_RMS


def silhouette(cell: np.ndarray) -> np.ndarray:
    """Opacité α (0..1) de la pièce en chaque pixel, sur une vignette de fond uni connu.

    Chaque pixel vaut ``C = αF + (1 − α)B`` : B est le fond (pourtour de la case)
    et F la couleur de la pièce, prise au pixel « plein » le plus proche (cœur de
    la pièce, loin de tout bord). α s'en déduit par projection. Les bords
    anticrénelés et les ombres reçoivent ainsi une opacité partielle, et les
    ouvertures de la pièce (les « yeux » de la couronne du roi) restent du fond :
    recomposé sur une autre case, le gabarit y montre la couleur de cette case.
    """
    cell = cell.astype(np.float64)
    bg = background(cell)
    dist = np.linalg.norm(cell - bg, axis=-1)
    labels = components(dist > PIECE_THRESHOLD, diagonal=True)
    if labels.max() == 0:
        return np.zeros((CELL, CELL), np.float64)
    # Plus grande tache seulement : écarte les coordonnées écrites dans les coins.
    piece = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))

    # Cœur de la pièce : pixels dont tous les voisins sont aussi dans la pièce.
    core = piece.copy()
    for dy, dx in NEIGHBOURS:
        core &= shift(piece, dy, dx)
    if not core.any():
        core = piece

    # Couleur de pièce F propagée du cœur vers les bords, voisin par voisin.
    known = core.copy()
    fg = np.where(core[..., None], cell, 0.0)
    for _ in range(MATTE_REACH):
        total = np.zeros_like(cell)
        count = np.zeros(cell.shape[:2])
        for dy, dx in NEIGHBOURS:
            neighbour_known = shift(known, dy, dx)
            total += shift(fg, dy, dx) * neighbour_known[..., None]
            count += neighbour_known
        new = ~known & (count > 0)
        fg[new] = total[new] / count[new][:, None]
        known |= new

    # α par projection de (C − B) sur (F − B), là où pièce et fond se distinguent.
    contrast = fg - bg
    denom = (contrast**2).sum(axis=-1)
    reliable = known & (denom > MIN_CONTRAST**2)
    alpha = np.where(
        reliable,
        ((cell - bg) * contrast).sum(axis=-1) / np.where(reliable, denom, 1.0),
        piece.astype(np.float64),
    )
    alpha[~known] = 0.0
    return np.clip(alpha, 0.0, 1.0)


@dataclass
class TemplateSet:
    """Gabarits de pièces (et coordonnées) mesurés sur une capture de la position initiale."""

    images: np.ndarray  # (n, CELL, CELL, 3)
    masks: np.ndarray  # (n, CELL, CELL), opacité 0..1
    labels: tuple[str, ...]  # symbole FEN de chaque gabarit
    theme: Theme
    glyphs: CoordinateGlyphs | None = None  # None : coordonnées non affichées

    @cached_property
    def _reduced_masks(self) -> np.ndarray:
        return reduce(self.masks)

    @cached_property
    def _reduced_premultiplied(self) -> np.ndarray:
        # Couleur prémultipliée αF = C − (1 − α)·B_calibration : recomposée sur un
        # fond B, elle redonne exactement C + (1 − α)(B − B_calibration).
        own_bg = background(self.images)[:, None, None, :]
        return reduce_rgb(self.images - (1 - self.masks[..., None]) * own_bg)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.glyphs is not None:
            extra = {"rank_glyphs": self.glyphs.ranks, "file_glyphs": self.glyphs.files}
        with path.open("wb") as fh:
            np.savez_compressed(
                fh,
                images=np.clip(self.images, 0, 255).round().astype(np.uint8),
                masks=(self.masks * 255).round().astype(np.uint8),
                labels=np.array(self.labels),
                theme=np.array(json.dumps(self.theme.to_dict())),
                **extra,
            )

    @classmethod
    def load(cls, path: str | Path) -> TemplateSet:
        with np.load(Path(path), allow_pickle=False) as data:
            glyphs = None
            if "rank_glyphs" in data:
                glyphs = CoordinateGlyphs(data["rank_glyphs"], data["file_glyphs"])
            return cls(
                images=data["images"].astype(np.float64),
                masks=data["masks"].astype(np.float64) / 255,
                labels=tuple(str(s) for s in data["labels"]),
                theme=Theme.from_dict(json.loads(str(data["theme"]))),
                glyphs=glyphs,
            )

    def errors(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Erreur quadratique moyenne de chaque vignette contre chaque gabarit et contre
        « case vide » : tableaux ``(k, n)`` et ``(k,)``.

        Le gabarit recomposé sur le fond B de la vignette vaut ``αT + (1 − α)B`` ;
        l'écart à la vignette C est ``D − αT + αB`` avec ``D = C − B``. Sa norme se
        développe en produits matriciels, sans construire les k × n recompositions.
        Tout est linéaire : la réduction (:func:`reduce`) s'applique à C, α et au
        gabarit prémultiplié αT sans fausser la recomposition.
        """
        k = cells.shape[0]
        n = len(self.labels)
        weights = reduce(WEIGHTS)
        pixels = weights.size
        sqrt_w = np.sqrt(weights).reshape(1, pixels)
        norm = 3 * weights.sum()

        cells = cells.astype(np.float64)
        bg = background(cells)  # (k, 3)
        reduced = reduce_rgb(cells).reshape(k, pixels, 3)
        d = (reduced - bg[:, None, :]) * sqrt_w[..., None]  # (k, P, 3), pondéré
        alpha = self._reduced_masks.reshape(n, pixels) * sqrt_w  # (n, P), pondéré
        alpha_t = self._reduced_premultiplied.reshape(n, pixels, 3) * sqrt_w[..., None]

        dd = (d**2).sum(axis=(1, 2))  # ‖D‖²            (k,)
        tt = (alpha_t**2).sum(axis=(1, 2))  # ‖αT‖²     (n,)
        aa = (alpha**2).sum(axis=1)  # Σα²             (n,)
        bb = (bg**2).sum(axis=1)  # ‖B‖²               (k,)
        d_at = d.reshape(k, -1) @ alpha_t.reshape(n, -1).T  # ⟨D, αT⟩   (k, n)
        d_ab = (d * bg[:, None, :]).sum(axis=-1) @ alpha.T  # ⟨D, αB⟩   (k, n)
        at_ab = bg @ (alpha[..., None] * alpha_t).sum(axis=1).T  # ⟨αT, αB⟩ (k, n)

        piece = dd[:, None] + tt[None] + aa[None] * bb[:, None] - 2 * d_at + 2 * d_ab - 2 * at_ab
        return np.maximum(piece, 0) / norm, dd / norm

    def classify(self, cells: np.ndarray) -> list[list[CellPrediction]]:
        """Classe ``8 × 8`` vignettes (ordre de l'écran) en pièces ou cases vides."""
        piece_err, empty_err = self.errors(cells.reshape(-1, CELL, CELL, 3))

        # Meilleure erreur par classe (plusieurs gabarits peuvent avoir le même symbole).
        classes = sorted(set(self.labels))
        columns = [
            piece_err[:, [i for i, s in enumerate(self.labels) if s == c]].min(axis=1)
            for c in classes
        ]
        rms = np.sqrt(np.stack([*columns, empty_err], axis=1))
        names = [*classes, EMPTY]

        order = np.argsort(rms, axis=1)
        predictions = []
        for idx in range(rms.shape[0]):
            best, second = rms[idx, order[idx, 0]], rms[idx, order[idx, 1]]
            predictions.append(
                CellPrediction(
                    label=names[order[idx, 0]],
                    confidence=max(0.0, 1.0 - float(best / max(second, 1e-6))),
                    rms=float(best),
                )
            )
        return [predictions[r * 8 : (r + 1) * 8] for r in range(8)]


def white_at_bottom(cells: np.ndarray) -> bool:
    """Sur la position initiale : les pièces les plus claires sont-elles en bas ?"""
    quarter = CELL // 4
    centre = cells[:, :, quarter:-quarter, quarter:-quarter].astype(np.float64)
    luminance = centre.mean(axis=(2, 3, 4))  # (8, 8)
    return bool(luminance[6:].mean() > luminance[:2].mean())


def calibrate(cells: np.ndarray, base_theme: Theme = CHESSCOM_GREEN) -> TemplateSet:
    """Construit les gabarits à partir des 64 vignettes de la position initiale."""
    flat = cells.astype(np.float64)

    # Rangées 3 à 6 : doivent être vides (fond uni, aux coordonnées près).
    middle = flat[2:6].reshape(-1, CELL, CELL, 3)
    spread = np.linalg.norm(middle - background(middle)[:, None, None, :], axis=-1)
    if (spread > PIECE_THRESHOLD).mean(axis=(1, 2)).max() > 0.05:
        raise CalibrationError(
            "Ce n'est pas la position initiale : les rangées centrales doivent être vides."
        )

    white_bottom = white_at_bottom(cells)
    start = chess.Board()
    images, masks, labels = [], [], []
    for row in (0, 1, 6, 7):
        for col in range(8):
            square = screen_to_square(row, col, white_bottom)
            piece = start.piece_at(square)
            assert piece is not None
            cell = flat[row, col]
            mask = silhouette(cell)
            if mask.mean() < 0.05:
                raise CalibrationError(
                    f"Pièce introuvable en {chess.square_name(square)} : "
                    "est-ce bien la position initiale ?"
                )
            images.append(cell)
            masks.append(mask)
            labels.append(piece.symbol())

    # Couleurs de fond mesurées sur les cases vides (claires : ligne + colonne paires).
    bgs = background(flat[2:6])
    parity = (np.add.outer(np.arange(2, 6), np.arange(8)) % 2) == 0
    light = tuple(int(v) for v in np.median(bgs[parity], axis=0))
    dark = tuple(int(v) for v in np.median(bgs[~parity], axis=0))
    theme = Theme(
        name=f"calibré ({base_theme.name})",
        light=(light, *base_theme.light),
        dark=(dark, *base_theme.dark),
    )
    glyphs = learn_glyphs(cells, white_bottom, theme)
    return TemplateSet(np.stack(images), np.stack(masks), tuple(labels), theme, glyphs)
