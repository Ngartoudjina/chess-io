"""Interface graphique PySide6 (EF-01, EF-03, EF-09).

Le fil principal ne fait que l'affichage : reconnaissance et moteur tournent
dans un fil de travail, et le moteur lui-même dans un processus séparé
(ENF-07). On ne propose jamais de coup sur une position douteuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import chess.svg
import numpy as np
from PySide6.QtCore import (
    QByteArray,
    QObject,
    QPoint,
    QRect,
    QSettings,
    QSize,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QGuiApplication, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRubberBand,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from chessio import __version__
from chessio.engine import DEFAULT_TIME, MAX_TIME, MIN_TIME, Analysis, Engine
from chessio.paths import templates_path
from chessio.position import validate
from chessio.recognize import Orientation, Recognition, recognize
from chessio.vision.board import BoardRegion, crop_board, load_image, locate_board, split_cells
from chessio.vision.templates import TemplateSet, calibrate

WARNING = (
    "Chessio conseille, il ne joue jamais à votre place.\n\n"
    "Chess.com interdit toute aide d'un programme ou d'un moteur (Stockfish compris) "
    "pendant une partie en cours. Utilisez Chessio uniquement pour les parties "
    "terminées, les parties contre les bots, les problèmes et l'étude."
)
# Meilleur coup en vert franc, suivants en teintes claires (le rendu SVG de Qt
# ignore la transparence écrite #RRGGBBAA).
ARROW_COLORS = ("#15781B", "#7FA8D9", "#E3B47C")


def qimage_to_rgb(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height, stride = image.width(), image.height(), image.bytesPerLine()
    buffer = np.frombuffer(image.constBits(), np.uint8, count=stride * height)
    return buffer.reshape(height, stride)[:, : width * 3].reshape(height, width, 3).copy()


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    height, width, _ = rgb.shape
    data = np.ascontiguousarray(rgb)
    return QImage(data.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class Worker(QObject):
    """Exécute la reconnaissance et le moteur hors du fil de l'interface."""

    engine_ready = Signal(str)
    recognized = Signal(object)
    analysed = Signal(object)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.engine: Engine | None = None

    @Slot()
    def start_engine(self) -> None:
        try:
            self.engine = Engine()
            self.engine_ready.emit(self.engine.name)
        except Exception as exc:  # noqa: BLE001 — remonté à l'utilisateur
            self.failed.emit(f"Moteur indisponible : {exc}")

    @Slot(object, object, object, object, object)
    def recognize(self, image, turn, templates, orientation, region) -> None:
        try:
            self.recognized.emit(
                recognize(image, turn, templates, orientation=orientation, region=region)
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    @Slot(str, float, int)
    def analyse(self, fen: str, seconds: float, multipv: int) -> None:
        if self.engine is None:
            self.failed.emit("Le moteur n'est pas encore prêt.")
            return
        try:
            self.analysed.emit(
                self.engine.analyse(chess.Board(fen), time_limit=seconds, multipv=multipv)
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    @Slot()
    def stop(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None


class ImageView(QLabel):
    """Affiche la capture ; en mode tracé, l'utilisateur y délimite le plateau."""

    region_drawn = Signal(object)  # BoardRegion en coordonnées de l'image

    def __init__(self) -> None:
        super().__init__(
            "Collez une capture (Ctrl+V), glissez-déposez-la ici\nou ouvrez un fichier (Ctrl+O)"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setStyleSheet("QLabel { border: 2px dashed #888; color: #666; }")
        self._pixmap: QPixmap | None = None
        self._band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin: QPoint | None = None
        self.drawing = False

    def set_image(self, rgb: np.ndarray) -> None:
        self._pixmap = QPixmap.fromImage(rgb_to_qimage(rgb))
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802 (API Qt)
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(
                self._pixmap.scaled(
                    self.size() - QSize(8, 8),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _shown_rect(self) -> QRect:
        shown = self.pixmap().size()
        x = (self.width() - shown.width()) // 2
        y = (self.height() - shown.height()) // 2
        return QRect(x, y, shown.width(), shown.height())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.drawing and self._pixmap is not None:
            self._origin = event.position().toPoint()
            self._band.setGeometry(QRect(self._origin, QSize()))
            self._band.show()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._origin is not None:
            self._band.setGeometry(QRect(self._origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._origin is None or self._pixmap is None:
            return
        rect = QRect(self._origin, event.position().toPoint()).normalized()
        self._origin = None
        self._band.hide()
        shown = self._shown_rect()
        scale = self._pixmap.width() / shown.width()
        side = max(rect.width(), rect.height()) * scale
        x = (rect.x() - shown.x()) * scale
        y = (rect.y() - shown.y()) * scale
        if side >= 64:
            self.region_drawn.emit(BoardRegion(int(max(0, x)), int(max(0, y)), int(side)))


class MainWindow(QMainWindow):
    request_recognize = Signal(object, object, object, object, object)
    request_analyse = Signal(str, float, int)
    request_stop = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Chessio {__version__} — meilleur coup")
        self.setAcceptDrops(True)
        self.image: np.ndarray | None = None
        self.region: BoardRegion | None = None  # zone tracée à la main
        self.recognition: Recognition | None = None
        self.templates = self._load_templates()

        self._build_ui()
        self._build_actions()
        self._start_worker()
        self._update_turn_banner()

    # --- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        self.view = ImageView()
        self.view.region_drawn.connect(self._on_region_drawn)

        self.board_svg = QSvgWidget()
        self.board_svg.setFixedSize(360, 360)

        self.turn_banner = QLabel()
        self.turn_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.white_btn = QPushButton("Blancs (W)")
        self.black_btn = QPushButton("Noirs (B)")
        self.turn_group = QButtonGroup(self)
        for button in (self.white_btn, self.black_btn):
            button.setCheckable(True)
            self.turn_group.addButton(button)
        self.white_btn.setChecked(True)
        self.turn_group.buttonClicked.connect(lambda _: self._on_turn_changed())
        turn_row = QHBoxLayout()
        turn_row.addWidget(self.white_btn)
        turn_row.addWidget(self.black_btn)

        self.orientation = QComboBox()
        self.orientation.addItem("Auto (d'après les coordonnées)", Orientation.AUTO)
        self.orientation.addItem("Blancs en bas", Orientation.WHITE_BOTTOM)
        self.orientation.addItem("Noirs en bas", Orientation.BLACK_BOTTOM)
        self.orientation.currentIndexChanged.connect(lambda _: self._run_recognition())

        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(MIN_TIME, MAX_TIME)
        self.seconds.setSingleStep(0.5)
        self.seconds.setValue(DEFAULT_TIME)
        self.seconds.setSuffix(" s")
        self.multipv = QSpinBox()
        self.multipv.setRange(1, 3)
        self.multipv.setValue(3)

        self.fen = QLineEdit()
        self.fen.setPlaceholderText("FEN (modifiable : roque, prise en passant…)")
        self.fen.returnPressed.connect(self._analyse_fen)
        self.analyse_btn = QPushButton("Analyser (Entrée)")
        self.analyse_btn.clicked.connect(self._analyse_fen)
        self.draw_btn = QPushButton("Tracer le plateau à la main")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self._on_draw_toggled)

        self.best = QLabel("—")
        self.best.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.best.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details = QTextBrowser()
        self.details.setMinimumHeight(110)

        settings = QFormLayout()
        settings.addRow("Orientation", self.orientation)
        settings.addRow("Temps de réflexion", self.seconds)
        settings.addRow("Coups proposés", self.multipv)
        settings_box = QGroupBox("Réglages")
        settings_box.setLayout(settings)

        right = QVBoxLayout()
        right.addWidget(self.turn_banner)
        right.addLayout(turn_row)
        right.addWidget(self.board_svg, alignment=Qt.AlignmentFlag.AlignHCenter)
        right.addWidget(self.best)
        right.addWidget(self.details)
        right.addWidget(settings_box)

        left = QVBoxLayout()
        left.addWidget(self.view, stretch=1)
        left.addWidget(self.draw_btn)
        left.addWidget(self.fen)
        left.addWidget(self.analyse_btn)

        root = QHBoxLayout()
        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=2)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self._show_board(chess.Board())

    def _build_actions(self) -> None:
        menu = self.menuBar().addMenu("&Fichier")
        for text, shortcut, slot in (
            ("&Ouvrir une capture…", QKeySequence.StandardKey.Open, self._open_file),
            ("&Coller une capture", QKeySequence.StandardKey.Paste, self._paste),
            ("Copier la &FEN", QKeySequence("Ctrl+Shift+C"), self._copy_fen),
            ("&Calibrer avec une capture de la position initiale…", None, self._calibrate),
            ("&Quitter", QKeySequence.StandardKey.Quit, self.close),
        ):
            action = QAction(text, self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            menu.addAction(action)
        QShortcut(QKeySequence("W"), self, activated=lambda: self._set_turn(chess.WHITE))
        QShortcut(QKeySequence("B"), self, activated=lambda: self._set_turn(chess.BLACK))

    def _start_worker(self) -> None:
        self.thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start_engine)
        self.request_recognize.connect(self.worker.recognize)
        self.request_analyse.connect(self.worker.analyse)
        self.request_stop.connect(self.worker.stop, Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.engine_ready.connect(
            lambda name: self.statusBar().showMessage(f"Moteur prêt : {name}")
        )
        self.worker.recognized.connect(self._on_recognized)
        self.worker.analysed.connect(self._on_analysed)
        self.worker.failed.connect(self._on_failed)
        self.statusBar().showMessage("Démarrage du moteur…")
        self.thread.start()

    # --- état ---------------------------------------------------------------

    @property
    def turn(self) -> chess.Color:
        return chess.WHITE if self.white_btn.isChecked() else chess.BLACK

    def _set_turn(self, color: chess.Color) -> None:
        (self.white_btn if color == chess.WHITE else self.black_btn).setChecked(True)
        self._on_turn_changed()

    def _update_turn_banner(self) -> None:
        white = self.turn == chess.WHITE
        self.turn_banner.setText(f"Trait aux {'BLANCS' if white else 'NOIRS'}")
        fg, bg = ("#222", "#f4f4f4") if white else ("#f4f4f4", "#222")
        self.turn_banner.setStyleSheet(
            f"font-size: 22px; font-weight: bold; padding: 6px; color: {fg}; "
            f"background: {bg}; border: 2px solid #888; border-radius: 6px;"
        )

    def _on_turn_changed(self) -> None:
        self._update_turn_banner()
        if self.image is not None:
            self._run_recognition()
        elif self.fen.text().strip():
            board = self._board_from_field()
            if board is not None:
                board.turn = self.turn
                self.fen.setText(board.fen())

    def _load_templates(self) -> TemplateSet | None:
        path = templates_path()
        return TemplateSet.load(path) if path.exists() else None

    # --- import de la capture (EF-01) --------------------------------------

    def _set_image(self, rgb: np.ndarray) -> None:
        self.image = rgb
        self.region = None
        self.view.set_image(rgb)
        self._run_recognition()

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une capture", str(Path.home()), "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self._set_image(load_image(path))

    def _paste(self) -> None:
        image = QGuiApplication.clipboard().image()
        if image.isNull():
            self.statusBar().showMessage("Le presse-papiers ne contient pas d'image.")
            return
        self._set_image(qimage_to_rgb(image))

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasImage() or event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        data = event.mimeData()
        if data.hasImage():
            self._set_image(qimage_to_rgb(QImage(data.imageData())))
        elif data.hasUrls():
            local = [u.toLocalFile() for u in data.urls() if u.isLocalFile()]
            if local:
                self._set_image(load_image(local[0]))

    def _on_draw_toggled(self, on: bool) -> None:
        self.view.drawing = on
        if on:
            self.statusBar().showMessage("Tracez un carré autour des 64 cases.")

    def _on_region_drawn(self, region: BoardRegion) -> None:
        self.region = region
        self.draw_btn.setChecked(False)
        self._run_recognition()

    # --- calibration --------------------------------------------------------

    def _calibrate(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Capture de la position initiale (thème habituel)",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg)",
        )
        if not path:
            return
        try:
            image = load_image(path)
            cells = split_cells(crop_board(image, locate_board(image)))
            templates = calibrate(cells)
        except ValueError as exc:
            QMessageBox.warning(self, "Calibration impossible", str(exc))
            return
        templates.save(templates_path())
        self.templates = templates
        self.statusBar().showMessage(f"Calibration enregistrée : {templates_path()}")
        if self.image is not None:
            self._run_recognition()

    # --- reconnaissance et analyse -----------------------------------------

    def _run_recognition(self) -> None:
        if self.image is None:
            return
        if self.templates is None:
            QMessageBox.information(
                self,
                "Calibration nécessaire",
                "Avant la première utilisation, ouvrez Fichier → Calibrer et donnez une "
                "capture de la position initiale, dans votre thème Chess.com habituel.",
            )
            return
        self.statusBar().showMessage("Reconnaissance…")
        self.best.setText("…")
        self.request_recognize.emit(
            self.image, self.turn, self.templates, self.orientation.currentData(), self.region
        )

    def _on_recognized(self, result: Recognition) -> None:
        self.recognition = result
        self.fen.setText(result.board.fen())
        self.statusBar().showMessage(self._recognition_summary())
        if result.ok:
            self._show_board(result.board, white_bottom=result.white_bottom)
            self._start_analysis(result.board)
        else:
            self._show_problems(result.board, result.issues, result.white_bottom)

    def _recognition_summary(self) -> str:
        result = self.recognition
        assert result is not None
        bottom = "Blancs" if result.white_bottom else "Noirs"
        return (
            f"Reconnaissance : {result.elapsed * 1000:.0f} ms · {bottom} en bas "
            f"(d'après : {result.orientation_source}) · "
        )

    def _board_from_field(self) -> chess.Board | None:
        try:
            return chess.Board(self.fen.text().strip())
        except ValueError as exc:
            self._on_failed(f"FEN incorrecte : {exc}")
            return None

    def _analyse_fen(self) -> None:
        board = self._board_from_field()
        if board is None:
            return
        if board.turn != self.turn:
            # La FEN saisie fait foi ; setChecked ne relance pas la reconnaissance.
            (self.white_btn if board.turn == chess.WHITE else self.black_btn).setChecked(True)
            self._update_turn_banner()
        white_bottom = self.recognition.white_bottom if self.recognition else True
        issues = validate(board)
        if issues:
            self._show_problems(board, issues, white_bottom)
            return
        self._show_board(board, white_bottom=white_bottom)
        self._start_analysis(board)

    def _start_analysis(self, board: chess.Board) -> None:
        self.best.setText(f"Calcul… ({self.seconds.value():.1f} s)")
        self.details.clear()
        self.request_analyse.emit(board.fen(), self.seconds.value(), self.multipv.value())

    def _on_analysed(self, analysis: Analysis) -> None:
        board = chess.Board(analysis.fen)
        white_bottom = self.recognition.white_bottom if self.recognition else True
        if analysis.outcome:
            self.best.setText(f"Partie terminée : {analysis.outcome}")
            self._show_board(board, white_bottom=white_bottom)
            return
        best = analysis.best
        assert best is not None
        self.best.setText(f"{best.san}  ({best.uci})   {best.evaluation}")
        # Meilleur coup dessiné en dernier, donc par-dessus les autres.
        arrows = [
            chess.svg.Arrow(line.move.from_square, line.move.to_square, color=color)
            for line, color in zip(analysis.lines, ARROW_COLORS, strict=False)
        ][::-1]
        self._show_board(board, white_bottom=white_bottom, arrows=arrows)
        rows = "".join(
            f"<p><b>{i}. {line.san}</b> ({line.uci}) — éval <b>{line.evaluation}</b>, "
            f"prof. {line.depth}<br><span style='color:#666'>{line.pv_san}</span></p>"
            for i, line in enumerate(analysis.lines, start=1)
        )
        self.details.setHtml(rows)
        recognition = self._recognition_summary() if self.recognition else ""
        self.statusBar().showMessage(f"{recognition}Analyse : {analysis.elapsed:.2f} s")

    def _show_problems(self, board: chess.Board, issues, white_bottom: bool) -> None:
        marked = {sq for issue in issues for sq in issue.squares}
        self._show_board(board, white_bottom=white_bottom, fill=dict.fromkeys(marked, "#e0303080"))
        self.best.setText("Aucun coup proposé : position à vérifier")
        self.details.setHtml(
            "<ul>" + "".join(f"<li>{issue}</li>" for issue in issues) + "</ul>"
            "<p>Corrigez la FEN ci-contre puis « Analyser », ou refaites la capture.</p>"
        )

    def _on_failed(self, message: str) -> None:
        self.best.setText("Erreur")
        self.details.setPlainText(message)
        self.statusBar().showMessage(message)

    def _show_board(self, board, *, white_bottom=True, arrows=(), fill=None) -> None:
        svg = chess.svg.board(
            board,
            orientation=chess.WHITE if white_bottom else chess.BLACK,
            arrows=list(arrows),
            fill=fill or {},
            size=360,
        )
        self.board_svg.load(QByteArray(svg.encode()))

    def _copy_fen(self) -> None:
        QGuiApplication.clipboard().setText(self.fen.text())
        self.statusBar().showMessage("FEN copiée.")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.request_stop.emit()
        self.thread.quit()
        self.thread.wait(3000)
        super().closeEvent(event)


def confirm_warning(parent: QWidget | None = None) -> bool:
    """Avertissement d'usage, affiché jusqu'à acceptation (section 2)."""
    settings = QSettings("chessio", "chessio")
    if settings.value("warning_accepted", False, type=bool):
        return True
    answer = QMessageBox.question(
        parent,
        "Cadre d'utilisation",
        WARNING + "\n\nAcceptez-vous ces conditions ?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if answer != QMessageBox.StandardButton.Yes:
        return False
    settings.setValue("warning_accepted", True)
    return True


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Chessio")
    if not confirm_warning():
        return 0
    window = MainWindow()
    window.resize(1100, 720)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
