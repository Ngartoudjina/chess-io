"""Travail hors du fil de l'interface : moteur, lecture des captures, calibration (ENF-07).

Chaque demande porte un numéro de génération. L'interface l'incrémente à chaque
nouvelle demande et l'inscrit dans ``Worker.latest`` : une analyse en cours
s'arrête dès qu'elle n'est plus la dernière demandée, et les résultats périmés
sont ignorés.
"""

from __future__ import annotations

import chess
from PySide6.QtCore import QObject, Signal, Slot

from chessio.engine import Engine
from chessio.paths import templates_path
from chessio.recognize import recognize
from chessio.vision.board import crop_board, locate_board, split_cells
from chessio.vision.templates import calibrate


class Worker(QObject):
    engine_ready = Signal(str)
    engine_failed = Signal(str)
    recognized = Signal(int, object)  # génération, Recognition
    progress = Signal(int, object)  # génération, Analysis provisoire
    analysed = Signal(int, object)  # génération, Analysis finale
    calibrated = Signal(object)  # TemplateSet
    failed = Signal(int, str)  # génération, message

    def __init__(self) -> None:
        super().__init__()
        self.engine: Engine | None = None
        self.latest = 0  # écrit par le fil de l'interface

    @Slot()
    def start_engine(self) -> None:
        try:
            self.engine = Engine()
            self.engine_ready.emit(self.engine.name)
        except Exception as exc:  # noqa: BLE001 — remonté à l'utilisateur
            self.engine_failed.emit(str(exc))

    @Slot(int, object, object, object, object, object, object)
    def recognize(self, generation, image, turn, default_turn, templates, orientation, region):
        if generation != self.latest:
            return
        try:
            result = recognize(
                image,
                turn,
                templates,
                orientation=orientation,
                region=region,
                default_turn=default_turn,
            )
            self.recognized.emit(generation, result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(generation, str(exc))

    @Slot(int, str, float, int)
    def analyse(self, generation: int, fen: str, seconds: float, multipv: int) -> None:
        if generation != self.latest:
            return
        if self.engine is None:
            self.failed.emit(generation, "Le moteur d'échecs n'est pas disponible.")
            return
        try:
            result = self.engine.analyse(
                chess.Board(fen),
                time_limit=seconds,
                multipv=multipv,
                on_update=lambda partial: self.progress.emit(generation, partial),
                should_stop=lambda: self.latest != generation,
            )
            if not result.stopped:
                self.analysed.emit(generation, result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(generation, str(exc))

    @Slot(int, object)
    def calibrate(self, generation: int, image) -> None:
        try:
            templates = calibrate(split_cells(crop_board(image, locate_board(image))))
            templates.save(templates_path())
            self.calibrated.emit(templates)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(generation, f"Calibration impossible : {exc}")

    @Slot()
    def stop(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None
