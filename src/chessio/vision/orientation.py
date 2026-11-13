"""Orientation du plateau : Blancs ou Noirs en bas de l'écran (EF-04).

Lire le plateau à l'envers donne une position légale mais absurde, et donc un
coup faux sans le moindre signal d'alerte. On décide, par ordre de fiabilité :

1. par les **coordonnées** que Chess.com écrit sur les cases : chiffres dans la
   colonne de gauche (« 8 » en haut si les Blancs sont en bas, « 1 » sinon),
   lettres dans la rangée du bas (« a » à gauche, ou « h ») ;
2. à défaut, par la **structure de pions** : les pions blancs sont d'ordinaire
   plus près du camp blanc que les pions noirs ;
3. en dernier recours, par la couleur au trait (on joue d'ordinaire avec son
   camp en bas), ce que l'interface signale comme une simple supposition.

Les coordonnées se lisent à partir de ~36 px par case (plateau de 290 px) ;
en deçà, elles ne font que quelques pixels de haut et l'on passe à l'étape 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from chessio.vision.board import screen_to_square
from chessio.vision.ops import background, blur
from chessio.vision.theme import Theme

# Zones des coordonnées dans une vignette de 64 pixels.
RANK_ZONE = (slice(0, 22), slice(0, 17))  # chiffre : en haut à gauche, colonne de gauche
FILE_ZONE = (slice(44, 64), slice(46, 64))  # lettre : en bas à droite, rangée du bas
INK_TOLERANCE = 35  # écart maximal (par canal) à la couleur d'écriture des coordonnées
MIN_INK = 6  # pixels d'encre en deçà desquels la coordonnée est jugée absente
MIN_PROBES = 4  # coordonnées lisibles nécessaires pour conclure
MIN_MARGIN = 0.08  # écart moyen de ressemblance exigé entre les deux hypothèses


@dataclass(frozen=True)
class CoordinateGlyphs:
    """Formes des coordonnées apprises à la calibration."""

    ranks: np.ndarray  # (8, h, w) : chiffre de chaque rangée, indice 0 = « 1 »
    files: np.ndarray  # (8, h, w) : lettre de chaque colonne, indice 0 = « a »


def _ink(cell: np.ndarray, zone: tuple[slice, slice], theme: Theme) -> np.ndarray:
    """Pixels de la zone écrits dans la couleur des coordonnées.

    Chess.com écrit les coordonnées dans la couleur de l'autre case (vert sur
    crème, crème sur vert) : ne garder que ces pixels écarte les pièces, qui
    sont blanches, grises ou noires.
    """
    cell = cell.astype(np.float64)
    bg = background(cell)
    light = np.asarray(theme.light[0], np.float64)
    dark = np.asarray(theme.dark[0], np.float64)
    color = dark if np.linalg.norm(bg - light) < np.linalg.norm(bg - dark) else light
    return (np.abs(cell[zone] - color) <= INK_TOLERANCE).all(axis=-1)


def _shape(ink: np.ndarray) -> np.ndarray:
    """Forme floutée, pour tolérer de petits décalages entre captures."""
    return blur(blur(ink.astype(np.float64)))


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float((a * b).sum() / norm) if norm > 0 else 0.0


def learn_glyphs(cells: np.ndarray, white_bottom: bool, theme: Theme) -> CoordinateGlyphs | None:
    """Apprend les 8 chiffres et les 8 lettres sur la capture de calibration.

    Renvoie ``None`` si les coordonnées ne sont pas affichées sur le plateau.
    """
    rank_shape = np.zeros((64, 64))[RANK_ZONE].shape
    file_shape = np.zeros((64, 64))[FILE_ZONE].shape
    ranks, files = np.zeros((8, *rank_shape)), np.zeros((8, *file_shape))
    found = 0
    for row in range(8):
        ink = _ink(cells[row, 0], RANK_ZONE, theme)
        if ink.sum() >= MIN_INK:
            ranks[chess.square_rank(screen_to_square(row, 0, white_bottom))] = _shape(ink)
            found += 1
    for col in range(8):
        ink = _ink(cells[7, col], FILE_ZONE, theme)
        if ink.sum() >= MIN_INK:
            files[chess.square_file(screen_to_square(7, col, white_bottom))] = _shape(ink)
            found += 1
    return CoordinateGlyphs(ranks, files) if found >= 12 else None


def from_coordinates(cells: np.ndarray, glyphs: CoordinateGlyphs, theme: Theme) -> bool | None:
    """Vrai si les Blancs sont en bas, faux si ce sont les Noirs, ``None`` si illisible."""
    scores: dict[bool, float] = {True: 0.0, False: 0.0}
    probes = 0
    for row in range(8):
        ink = _ink(cells[row, 0], RANK_ZONE, theme)
        if ink.sum() < MIN_INK:
            continue
        probes += 1
        for white_bottom in (True, False):
            rank = chess.square_rank(screen_to_square(row, 0, white_bottom))
            scores[white_bottom] += _similarity(_shape(ink), glyphs.ranks[rank])
    for col in range(8):
        ink = _ink(cells[7, col], FILE_ZONE, theme)
        if ink.sum() < MIN_INK:
            continue
        probes += 1
        for white_bottom in (True, False):
            file = chess.square_file(screen_to_square(7, col, white_bottom))
            scores[white_bottom] += _similarity(_shape(ink), glyphs.files[file])
    if probes < MIN_PROBES:
        return None
    margin = (scores[True] - scores[False]) / probes
    return None if abs(margin) < MIN_MARGIN else margin > 0


def from_pawns(labels: list[list[str]]) -> bool | None:
    """D'après la structure de pions (``labels`` en ordre de l'écran, symboles FEN).

    Vus côté Blancs, les pions blancs sont en moyenne plus bas à l'écran que les
    pions noirs. Il faut au moins deux pions de chaque camp et un écart net.
    """
    white = [row for row in range(8) for col in range(8) if labels[row][col] == "P"]
    black = [row for row in range(8) for col in range(8) if labels[row][col] == "p"]
    if len(white) < 2 or len(black) < 2:
        return None
    gap = float(np.mean(white) - np.mean(black))  # > 0 : pions blancs plus bas
    return None if abs(gap) < 1.0 else gap > 0
