"""Chaîne de reconnaissance : capture + couleur au trait → position validée.

C'est le seul point de passage entre vision et moteur : il produit un
``chess.Board`` et la liste des problèmes qui empêchent de l'analyser.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import chess
import numpy as np

from chessio.position import Issue, build_board, parse_turn, validate
from chessio.vision.board import (
    BoardRegion,
    crop_board,
    locate_board,
    screen_to_square,
    split_cells,
)
from chessio.vision.orientation import from_coordinates, from_pawns
from chessio.vision.templates import EMPTY, CellPrediction, TemplateSet


class Orientation(Enum):
    AUTO = "auto"  # coordonnées, à défaut structure de pions, à défaut camp au trait en bas
    WHITE_BOTTOM = "w"
    BLACK_BOTTOM = "b"


# Comment l'orientation a été établie, de la plus sûre à la plus fragile.
SOURCE_SETTING = "réglage"
SOURCE_COORDINATES = "coordonnées"
SOURCE_PAWNS = "structure de pions"
SOURCE_TURN = "camp au trait supposé en bas"


@dataclass(frozen=True)
class Recognition:
    board: chess.Board
    region: BoardRegion
    white_bottom: bool
    orientation_source: str
    predictions: dict[chess.Square, CellPrediction]
    issues: list[Issue] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def doubtful(self) -> tuple[chess.Square, ...]:
        return tuple(sq for sq, p in sorted(self.predictions.items()) if p.doubtful)

    @property
    def ok(self) -> bool:
        """Vrai si l'on peut proposer un coup sans réserve."""
        return not self.issues

    @property
    def orientation_assumed(self) -> bool:
        """Vrai si l'orientation n'est qu'une supposition (ni réglée ni lue)."""
        return self.orientation_source == SOURCE_TURN


def _orientation(
    cells: np.ndarray,
    labels: list[list[str]],
    templates: TemplateSet,
    orientation: Orientation,
    turn: chess.Color,
) -> tuple[bool, str]:
    if orientation is Orientation.WHITE_BOTTOM:
        return True, SOURCE_SETTING
    if orientation is Orientation.BLACK_BOTTOM:
        return False, SOURCE_SETTING
    if templates.glyphs is not None:
        detected = from_coordinates(cells, templates.glyphs, templates.theme)
        if detected is not None:
            return detected, SOURCE_COORDINATES
    detected = from_pawns(labels)
    if detected is not None:
        return detected, SOURCE_PAWNS
    return turn == chess.WHITE, SOURCE_TURN


def recognize(
    image: np.ndarray,
    turn: str | chess.Color,
    templates: TemplateSet,
    *,
    orientation: Orientation = Orientation.AUTO,
    region: BoardRegion | None = None,
) -> Recognition:
    """Reconnaît la position d'une capture RGB. ``region`` force la zone du plateau."""
    start = time.perf_counter()
    color = parse_turn(turn)
    if region is None:
        region = locate_board(image, templates.theme)
    cells = split_cells(crop_board(image, region))

    # La classification se fait dans l'ordre de l'écran : elle ne dépend pas de
    # l'orientation, qui n'intervient que pour nommer les cases.
    grid = templates.classify(cells)
    labels = [[prediction.label for prediction in row] for row in grid]
    white_bottom, source = _orientation(cells, labels, templates, orientation, color)

    predictions: dict[chess.Square, CellPrediction] = {}
    placement = chess.Board(None)
    for row in range(8):
        for col in range(8):
            square = screen_to_square(row, col, white_bottom)
            prediction = grid[row][col]
            predictions[square] = prediction
            if prediction.label != EMPTY:
                placement.set_piece_at(square, chess.Piece.from_symbol(prediction.label))

    board = build_board(placement.board_fen(), color)
    issues = validate(board)
    doubtful = tuple(sq for sq in chess.SQUARES if predictions[sq].doubtful)
    if doubtful:
        issues.insert(0, Issue("Cases mal lues, à vérifier", doubtful))
    if source == SOURCE_TURN:
        # Lu à l'envers, le plateau donne une position légale mais un coup faux :
        # une orientation simplement supposée ne suffit pas pour conseiller un coup.
        issues.append(
            Issue(
                "Orientation du plateau non lue (coordonnées absentes ou illisibles) : "
                "indiquez si les Blancs ou les Noirs sont en bas"
            )
        )
    return Recognition(
        board=board,
        region=region,
        white_bottom=white_bottom,
        orientation_source=source,
        predictions=predictions,
        issues=issues,
        elapsed=time.perf_counter() - start,
    )
