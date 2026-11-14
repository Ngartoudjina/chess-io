"""Travail hors du fil de l'interface : moteur, lecture des captures, calibration (ENF-07).

Chaque demande porte un numéro de génération. L'interface l'incrémente à chaque
nouvelle demande et l'inscrit dans ``Worker.latest`` : une analyse en cours
s'arrête dès qu'elle n'est plus la dernière demandée, et les résultats périmés
sont ignorés.
"""

from __future__ import annotations

import chess
from PySide6.QtCore import QObject, Signal, Slot

from chessio.capture import CaptureError, capture_phone
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
    phone_captured = Signal(int, object)  # génération, image RGB
    failed = Signal(int, str)  # génération, message

    def __init__(self, engine_choice: str | None = None) -> None:
        super().__init__()
        self.engine: Engine | None = None
        self.engine_choice = engine_choice  # None : Stockfish ; « maison » : moteur maison
        self.latest = 0  # écrit par le fil de l'interface

    @Slot()
    def start_engine(self) -> None:
        try:
            self.engine = Engine.open(self.engine_choice)
            self.engine_ready.emit(self.engine.name)
        except Exception as exc:  # noqa: BLE001 — remonté à l'utilisateur
            self.engine_failed.emit(str(exc))

    @Slot(object)
    def switch_engine(self, choice: str | None) -> None:
        """Change de moteur (EF-10) : arrête l'actuel, lance le nouveau."""
        self.stop()
        self.engine_choice = choice
        self.start_engine()

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
                time_limit=seconds if seconds > 0 else None,  # 0 : illimitée
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

    @Slot(int, str)
    def capture_phone(self, generation: int, serial: str) -> None:
        try:
            self.phone_captured.emit(generation, capture_phone(serial))
        except CaptureError as exc:
            self.failed.emit(generation, str(exc))

    @Slot()
    def stop(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None
