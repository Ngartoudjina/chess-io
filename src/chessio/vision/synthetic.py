"""Rendu synthétique de captures d'échiquier (tests, puis jeu de données du lot L2).

Les pièces sont dessinées avec les glyphes d'échecs de la police DejaVu Sans,
sur les couleurs d'un thème, dans une « page » de couleur neutre.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chessio.vision.theme import CHESSCOM_GREEN, Theme

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)
BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
)
PAGE_COLOR = (49, 46, 43)  # fond sombre de Chess.com
# Glyphe plein (♚♛♜♝♞♟) pour les deux camps ; seule la couleur change.
GLYPHS = {
    chess.KING: "♚",
    chess.QUEEN: "♛",
    chess.ROOK: "♜",
    chess.BISHOP: "♝",
    chess.KNIGHT: "♞",
    chess.PAWN: "♟",
}


@lru_cache(maxsize=16)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise FileNotFoundError("Police DejaVu Sans introuvable (paquet fonts-dejavu-core)")


def render_board(
    board: chess.Board,
    *,
    square: int = 64,
    white_bottom: bool = True,
    theme: Theme = CHESSCOM_GREEN,
    margin: tuple[int, int, int, int] = (40, 30, 120, 50),
    highlight: tuple[chess.Square, ...] = (),
    coordinates: bool = True,
) -> np.ndarray:
    """Capture RGB simulée : plateau de ``8·square`` pixels dans une page.

    ``margin`` = (gauche, haut, droite, bas) ; ``highlight`` surligne des cases
    comme le dernier coup de Chess.com ; ``coordinates`` écrit, comme Chess.com,
    le chiffre des rangées dans la colonne de gauche (en haut à gauche de la
    case) et la lettre des colonnes dans la rangée du bas (en bas à droite),
    dans la couleur de l'autre case.
    """
    left, top, right, bottom = margin
    size = 8 * square
    image = Image.new("RGB", (left + size + right, top + size + bottom), PAGE_COLOR)
    draw = ImageDraw.Draw(image)
    font = _font(int(square * 0.82))
    label_font = _font(max(6, int(square * 0.2)), bold=True)

    for sq in chess.SQUARES:
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        col, row = (file, 7 - rank) if white_bottom else (7 - file, rank)
        light = (file + rank) % 2 == 1
        palette = theme.light if light else theme.dark
        color = palette[-1] if sq in highlight else palette[0]
        x0, y0 = left + col * square, top + row * square
        draw.rectangle((x0, y0, x0 + square - 1, y0 + square - 1), fill=color)

        if coordinates:
            ink = theme.dark[0] if light else theme.light[0]
            if col == 0:
                draw.text(
                    (x0 + 0.06 * square, y0 + 0.04 * square),
                    str(rank + 1),
                    font=label_font,
                    fill=ink,
                    anchor="lt",
                )
            if row == 7:
                draw.text(
                    (x0 + 0.95 * square, y0 + 0.97 * square),
                    chess.FILE_NAMES[file],
                    font=label_font,
                    fill=ink,
                    anchor="rb",
                )

        piece = board.piece_at(sq)
        if piece is None:
            continue
        fill, stroke = ((250, 250, 250), (40, 40, 40))
        if piece.color == chess.BLACK:
            fill, stroke = ((35, 35, 35), (90, 90, 90))
        draw.text(
            (x0 + square / 2, y0 + square / 2),
            GLYPHS[piece.piece_type],
            font=font,
            fill=fill,
            anchor="mm",
            stroke_width=max(1, square // 32),
            stroke_fill=stroke,
        )
    return np.asarray(image, dtype=np.uint8).copy()


def random_position(rng: np.random.Generator, plies: int | None = None) -> chess.Board:
    """Position plausible : une partie aléatoire de ``plies`` demi-coups."""
    board = chess.Board()
    for _ in range(int(rng.integers(10, 80)) if plies is None else plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(moves[int(rng.integers(len(moves)))])
    return board
