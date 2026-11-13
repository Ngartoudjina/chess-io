"""Chaîne de reconnaissance : capture → position validée, prête pour le moteur.

C'est le seul point de passage entre vision et moteur : il produit un
``chess.Board`` et la liste des problèmes qui empêchent de l'analyser.

Outre le placement des pièces, la capture renseigne sur le contexte :

* l'orientation (coordonnées affichées, à défaut structure de pions) ;
* le trait : le dernier coup surligné dit qui vient de jouer ; à défaut, un roi
  en échec est forcément celui du camp au trait ;
* la prise en passant : possible si le dernier coup est un pion avancé de deux cases.
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
from chessio.vision.lastmove import last_move
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

# Comment le trait a été établi.
TURN_INPUT = "choisi"
TURN_LAST_MOVE = "dernier coup joué"
TURN_CHECK = "roi en échec"
TURN_DEFAULT = "non détecté"


@dataclass(frozen=True)
class Recognition:
    board: chess.Board
    region: BoardRegion
    white_bottom: bool
    orientation_source: str
    predictions: dict[chess.Square, CellPrediction]
    issues: list[Issue] = field(default_factory=list)
    elapsed: float = 0.0
    turn_source: str = TURN_INPUT
    last_move: chess.Move | None = None  # dernier coup lu sur la capture (surlignage)
    notes: tuple[str, ...] = ()  # remarques non bloquantes

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

    @property
    def turn_detected(self) -> bool:
        return self.turn_source in (TURN_LAST_MOVE, TURN_CHECK)


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


def turn_from_check(placement: chess.Board) -> chess.Color | None:
    """Si un seul roi est en échec, c'est forcément au camp de ce roi de jouer."""
    white_king, black_king = placement.king(chess.WHITE), placement.king(chess.BLACK)
    if white_king is None or black_king is None:
        return None
    white_checked = placement.is_attacked_by(chess.BLACK, white_king)
    black_checked = placement.is_attacked_by(chess.WHITE, black_king)
    if white_checked == black_checked:
        return None
    return chess.WHITE if white_checked else chess.BLACK


def _name(color: chess.Color) -> str:
    return "Blancs" if color == chess.WHITE else "Noirs"


def recognize(
    image: np.ndarray,
    turn: str | chess.Color | None,
    templates: TemplateSet,
    *,
    orientation: Orientation = Orientation.AUTO,
    region: BoardRegion | None = None,
    default_turn: str | chess.Color = chess.WHITE,
) -> Recognition:
    """Reconnaît la position d'une capture RGB.

    ``turn`` impose la couleur au trait ; ``None`` la fait déduire de la capture
    (dernier coup surligné, roi en échec), ``default_turn`` servant à défaut.
    ``region`` force la zone du plateau (tracé à la main).
    """
    start = time.perf_counter()
    chosen = None if turn is None else parse_turn(turn)
    fallback = parse_turn(default_turn)
    if region is None:
        region = locate_board(image, templates.theme)
    cells = split_cells(crop_board(image, region))

    # La classification se fait dans l'ordre de l'écran : elle ne dépend pas de
    # l'orientation, qui n'intervient que pour nommer les cases.
    grid = templates.classify(cells)
    labels = [[prediction.label for prediction in row] for row in grid]

    # Dernier coup surligné : la pièce arrivée dit qui vient de jouer.
    moved = last_move(cells, labels, templates.theme)
    mover = None
    if moved is not None:
        (to_row, to_col) = moved[1]
        mover = chess.WHITE if labels[to_row][to_col].isupper() else chess.BLACK
    hinted = None if mover is None else not mover

    assumed = chosen if chosen is not None else (hinted if hinted is not None else fallback)
    white_bottom, source = _orientation(cells, labels, templates, orientation, assumed)

    predictions: dict[chess.Square, CellPrediction] = {}
    placement = chess.Board(None)
    for row in range(8):
        for col in range(8):
            square = screen_to_square(row, col, white_bottom)
            prediction = grid[row][col]
            predictions[square] = prediction
            if prediction.label != EMPTY:
                placement.set_piece_at(square, chess.Piece.from_symbol(prediction.label))

    # Trait : choix explicite, sinon dernier coup, sinon roi en échec, sinon défaut.
    detected, detected_source = hinted, TURN_LAST_MOVE
    if detected is None:
        detected, detected_source = turn_from_check(placement), TURN_CHECK
    notes: list[str] = []
    if chosen is not None:
        color, turn_source = chosen, TURN_INPUT
        if detected is not None and detected != chosen:
            notes.append(
                f"La capture indique plutôt le trait aux {_name(detected)} ({detected_source})."
            )
    elif detected is not None:
        color, turn_source = detected, detected_source
    else:
        color, turn_source = fallback, TURN_DEFAULT

    move = None
    en_passant = "-"
    if moved is not None:
        from_square = screen_to_square(*moved[0], white_bottom)
        to_square = screen_to_square(*moved[1], white_bottom)
        move = chess.Move(from_square, to_square)
        piece = placement.piece_at(to_square)
        double_step = abs(chess.square_rank(to_square) - chess.square_rank(from_square)) == 2
        if (
            piece is not None
            and piece.piece_type == chess.PAWN
            and piece.color != color
            and double_step
            and chess.square_file(to_square) == chess.square_file(from_square)
        ):
            en_passant = chess.square_name((from_square + to_square) // 2)

    board = build_board(placement.board_fen(), color, en_passant=en_passant)
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
        turn_source=turn_source,
        last_move=move,
        notes=tuple(notes),
    )
