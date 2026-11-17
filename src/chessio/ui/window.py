"""Fenêtre principale : coller une capture, lire le meilleur coup."""

from __future__ import annotations

import time

import chess
import chess.pgn
import numpy as np
from PySide6.QtCore import QPoint, QRect, QSettings, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRubberBand,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chessio import __version__
from chessio.capture import phones
from chessio.engine import DEFAULT_TIME, HOMEMADE, Analysis
from chessio.paths import templates_path
from chessio.position import validate
from chessio.recognize import (
    TURN_CHECK,
    TURN_DEFAULT,
    TURN_INPUT,
    TURN_LAST_MOVE,
    Orientation,
    Recognition,
    turn_from_check,
)
from chessio.ui.notation import figurine, figurine_line
from chessio.ui.screen import ScreenCapturer
from chessio.ui.style import ARROW_NAMES, PALETTE, app_icon, icon, piece_icon
from chessio.ui.updates import Updater, share
from chessio.ui.widgets import (
    Banner,
    BoardPanel,
    ElidedLabel,
    LineRow,
    Segmented,
    Toast,
    WdlBar,
    caption,
    card,
)
from chessio.ui.worker import Worker
from chessio.vision.board import BoardRegion, load_image
from chessio.vision.classify import load_model
from chessio.vision.templates import TemplateSet

WARNING = (
    "Chessio conseille, il ne joue jamais à votre place.\n\n"
    "Chess.com interdit toute aide d'un programme ou d'un moteur (Stockfish compris) "
    "pendant une partie en cours. Utilisez Chessio uniquement pour les parties "
    "terminées, les parties contre les bots, les problèmes et l'étude."
)
UNLIMITED = 0.0  # durée d'analyse : sans limite, jusqu'à « Arrêter »
LINE_COLORS = ("#f59e0b", "#38bdf8", "#c084fc")  # mêmes teintes que les flèches
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
PIECE_NAMES = {
    "K": "Roi blanc",
    "Q": "Dame blanche",
    "R": "Tour blanche",
    "B": "Fou blanc",
    "N": "Cavalier blanc",
    "P": "Pion blanc",
    "k": "Roi noir",
    "q": "Dame noire",
    "r": "Tour noire",
    "b": "Fou noir",
    "n": "Cavalier noir",
    "p": "Pion noir",
}


def qimage_to_rgb(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height, stride = image.width(), image.height(), image.bytesPerLine()
    buffer = np.frombuffer(image.constBits(), np.uint8, count=stride * height)
    return buffer.reshape(height, stride)[:, : width * 3].reshape(height, width, 3).copy()


def rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    height, width, _ = rgb.shape
    data = np.ascontiguousarray(rgb)
    image = QImage(data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def seconds_text(seconds: float) -> str:
    return f"{seconds:.1f}".replace(".", ",") + " s"


class SelectableImage(QLabel):
    """Image sur laquelle on trace un rectangle à la souris."""

    selected = Signal(object)  # BoardRegion en coordonnées de l'image

    def __init__(self, rgb: np.ndarray, max_size: QSize) -> None:
        super().__init__()
        self._full = rgb_to_pixmap(rgb)
        shown = self._full.scaled(
            max_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(shown)
        self.setFixedSize(shown.size())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._scale = self._full.width() / shown.width()
        self._band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin: QPoint | None = None

    def mousePressEvent(self, event) -> None:  # noqa: N802 (API Qt)
        self._origin = event.position().toPoint()
        self._band.setGeometry(QRect(self._origin, QSize()))
        self._band.show()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._origin is not None:
            self._band.setGeometry(QRect(self._origin, event.position().toPoint()).normalized())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._origin is None:
            return
        rect = QRect(self._origin, event.position().toPoint()).normalized()
        self._origin = None
        side = max(rect.width(), rect.height())
        if side * self._scale >= 64:
            self.selected.emit(
                BoardRegion(
                    int(rect.x() * self._scale),
                    int(rect.y() * self._scale),
                    int(side * self._scale),
                )
            )


class RegionDialog(QDialog):
    """Délimiter le plateau à la main quand la détection automatique échoue (EF-02)."""

    def __init__(self, rgb: np.ndarray, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Délimiter le plateau")
        self.region: BoardRegion | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        hint = QLabel("Tracez un carré autour des 64 cases de l'échiquier (sans les bords).")
        hint.setObjectName("Muted")
        image = SelectableImage(rgb, QSize(960, 640))
        image.selected.connect(self._on_selected)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Utiliser ce cadre")
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Annuler")
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(hint)
        layout.addWidget(image, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._buttons)

    def _on_selected(self, region: BoardRegion) -> None:
        self.region = region
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)


class MainWindow(QMainWindow):
    request_recognize = Signal(int, object, object, object, object, object, object)
    request_analyse = Signal(int, str, float, int)
    request_calibrate = Signal(int, object)
    request_phone = Signal(int, str)
    request_engine = Signal(object)
    request_stop = Signal()

    def __init__(self, offer_capture: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Chessio — meilleur coup")
        # Mode capture (parties contre les bots) : source et camp du joueur.
        self.capture_source: str | None = None  # « phone » ou « screen »
        self.phone_serial = ""
        self.screen_zone: BoardRegion | None = None  # zone de l'écran contenant l'échiquier
        self.my_color: chess.Color | None = None
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.setMinimumSize(1000, 680)

        self.templates = self._load_templates()
        self.image: np.ndarray | None = None
        self.region: BoardRegion | None = None
        self.recognition: Recognition | None = None
        self.board: chess.Board | None = None  # position affichée, éventuellement corrigée
        self.white_bottom = True
        self.lastmove: chess.Move | None = None
        self.doubtful: set[chess.Square] = set()
        self.verified: set[chess.Square] = set()
        self.turn_explicit = False
        self.turn_source = TURN_DEFAULT  # comment le trait affiché a été établi
        self.analysis: Analysis | None = None
        self.selected_line = 0
        self.generation = 0
        self.engine_ready = False
        self.calibrating = False
        self._pending_analysis = False
        self._settings = QSettings("chessio", "chessio")
        self._analysis_started = 0.0
        self._analysis_seconds = DEFAULT_TIME

        self._build()
        self._build_actions()
        self.updater = Updater(self)
        self._start_worker()
        self.screen = ScreenCapturer(self)
        self.screen.captured.connect(self._on_screen_captured)
        self.screen.failed.connect(lambda message: self._on_failed(self.generation, message))
        self._show_empty()
        if offer_capture:
            QTimer.singleShot(500, self._offer_capture_mode)
            if Updater.check_on_start():
                QTimer.singleShot(3000, lambda: self.updater.check(interactive=False))

    # --- construction ---------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        column = QVBoxLayout(root)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(18, 16, 18, 16)
        body.setSpacing(18)
        body.addLayout(self._build_left(), stretch=1)
        body.addWidget(self._build_right())
        column.addLayout(body, stretch=1)
        self.setCentralWidget(root)
        self.toast = Toast(root)

    def _header_button(self, name: str, text: str, tip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("HeaderButton")
        button.setIcon(icon(name))
        button.setIconSize(QSize(18, 18))
        button.setText(text)
        button.setToolTip(tip)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("Header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 10, 12, 10)
        layout.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(36, 36))
        logo.setStyleSheet("background: transparent;")
        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("Chessio")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Le meilleur coup, hors ligne")
        subtitle.setObjectName("AppSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        self.engine_chip = ElidedLabel()
        self.engine_chip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.engine_chip.set_full_text("●  Moteur : démarrage…")
        self.engine_chip.setObjectName("AppSubtitle")

        paste = self._header_button("paste", "Coller", "Coller une capture (Ctrl+V)")
        paste.clicked.connect(self._paste)
        open_ = self._header_button("open", "Ouvrir", "Ouvrir une image (Ctrl+O)")
        open_.clicked.connect(self._open_file)
        play = self._header_button(
            "play", "Jouer contre l'IA", "Jouer une partie contre Chessio (Ctrl+J)"
        )
        play.clicked.connect(self._open_play)
        self.menu_button = self._header_button("menu", "", "Plus d'actions")
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        # Les titres ne se tassent jamais ; c'est l'état du moteur qui cède la place.
        for label in (title, subtitle):
            label.setMinimumWidth(label.sizeHint().width())
        self.engine_chip.setMinimumWidth(40)
        layout.addWidget(logo)
        layout.addLayout(titles)
        layout.addWidget(self.engine_chip, stretch=1)
        layout.addSpacing(12)
        self.capture_button = QPushButton(icon("target", PALETTE["accent_text"]), "  Capturer (F5)")
        self.capture_button.setObjectName("Primary")
        self.capture_button.setToolTip(
            "Capture la partie (téléphone ou écran) et calcule le coup — parties contre les bots"
        )
        self.capture_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.capture_button.clicked.connect(self._capture)
        layout.addWidget(self.capture_button)
        for button in (play, paste, open_, self.menu_button):
            layout.addWidget(button)
        return header

    def _build_left(self) -> QVBoxLayout:
        left = QVBoxLayout()
        left.setSpacing(10)
        self.banner = Banner()
        self.panel = BoardPanel()
        self.panel.board.square_clicked.connect(self._on_square_clicked)
        self.panel.hint.paste_requested.connect(self._paste)
        self.panel.hint.open_requested.connect(self._open_file)
        self.panel.hint.play_requested.connect(self._open_play)
        self.status = QLabel()
        self.status.setObjectName("Status")

        fen_row = QHBoxLayout()
        fen_row.setSpacing(8)
        self.fen_edit = QLineEdit()
        self.fen_edit.setPlaceholderText("FEN : tapez ou collez une position, puis Entrée")
        self.fen_edit.returnPressed.connect(self._load_fen)
        copy = QPushButton(icon("copy"), "")
        copy.setToolTip("Copier la FEN")
        copy.clicked.connect(self._copy_fen)
        fen_row.addWidget(self.fen_edit, stretch=1)
        fen_row.addWidget(copy)

        left.addWidget(self.banner)
        left.addWidget(self.panel, stretch=1)
        left.addWidget(self.status)
        left.addLayout(fen_row)
        return left

    def _build_right(self) -> QScrollArea:
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(12)
        column.addWidget(self._build_turn_card())
        column.addWidget(self._build_best_card())
        column.addWidget(self._build_lines_card())
        column.addWidget(self._build_capture_card())
        column.addWidget(self._build_settings_card())
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(380)
        return scroll

    def _build_turn_card(self) -> QFrame:
        frame, layout = card()
        layout.addWidget(caption("Au trait"))
        self.turn = Segmented(
            [("○   Blancs", chess.WHITE), ("●   Noirs", chess.BLACK)],
            object_names=["TurnWhite", "TurnBlack"],
        )
        self.turn.changed.connect(self._on_turn_clicked)
        self.turn_hint = QLabel()
        self.turn_hint.setWordWrap(True)
        layout.addWidget(self.turn)
        layout.addWidget(self.turn_hint)
        return frame

    def _build_best_card(self) -> QFrame:
        frame, layout = card(spacing=8)
        layout.addWidget(caption("Meilleur coup"))
        row = QHBoxLayout()
        self.best_label = QLabel("—")
        self.best_label.setObjectName("BestMove")
        self.best_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.eval_chip = QLabel("")
        self.eval_chip.setObjectName("EvalChip")
        self.eval_chip.hide()
        row.addWidget(self.best_label)
        row.addStretch(1)
        row.addWidget(self.eval_chip, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.best_detail = QLabel("")
        self.best_detail.setObjectName("Muted")
        self.best_detail.setWordWrap(True)
        self.wdl_bar = WdlBar()
        self.wdl_label = QLabel("")
        self.wdl_label.setObjectName("Hint")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        buttons = QHBoxLayout()
        self.deeper_button = QPushButton(icon("deeper"), "  Analyse illimitée")
        self.deeper_button.setToolTip(
            "Le moteur approfondit sans limite de temps, en direct, jusqu'à « Arrêter » (Espace)"
        )
        self.deeper_button.clicked.connect(lambda: self._start_analysis(UNLIMITED))
        self.stop_button = QPushButton(icon("stop"), "  Arrêter")
        self.stop_button.setToolTip("Arrêter l'analyse (Échap)")
        self.stop_button.clicked.connect(self._stop_analysis)
        self.stop_button.hide()
        for button in (self.deeper_button, self.stop_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            buttons.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(self.best_detail)
        layout.addSpacing(4)
        layout.addWidget(self.wdl_bar)
        layout.addWidget(self.wdl_label)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)
        return frame

    def _build_lines_card(self) -> QFrame:
        frame, layout = card(spacing=8)
        layout.addWidget(caption("Coups possibles"))
        self.line_rows = []
        for index in range(3):
            row = LineRow(index)
            row.clicked.connect(self._on_line_clicked)
            row.hide()
            self.line_rows.append(row)
            layout.addWidget(row)
        self.lines_empty = QLabel("Les meilleures suites s'afficheront ici.")
        self.lines_empty.setObjectName("Hint")
        layout.addWidget(self.lines_empty)
        return frame

    def _build_capture_card(self) -> QFrame:
        frame, layout = card(spacing=10)
        layout.addWidget(caption("Capture"))
        row = QHBoxLayout()
        row.setSpacing(12)
        self.thumbnail = QLabel()
        self.thumbnail.setObjectName("Thumbnail")
        self.thumbnail.setFixedSize(96, 96)
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info = QVBoxLayout()
        info.setSpacing(2)
        self.capture_status = QLabel("Aucune capture")
        self.capture_status.setStyleSheet("font-weight: 700;")
        self.capture_detail = QLabel("")
        self.capture_detail.setObjectName("Hint")
        self.capture_detail.setWordWrap(True)
        info.addWidget(self.capture_status)
        info.addWidget(self.capture_detail)
        info.addStretch(1)
        row.addWidget(self.thumbnail)
        row.addLayout(info, stretch=1)
        buttons = QHBoxLayout()
        self.crop_button = QPushButton(icon("crop"), "  Délimiter…")
        self.crop_button.setToolTip("Tracer le plateau à la main si la détection échoue")
        self.crop_button.clicked.connect(self._draw_region)
        self.flip_button = QPushButton(icon("flip"), "  Retourner")
        self.flip_button.setToolTip("Lire l'échiquier dans l'autre sens (F)")
        self.flip_button.clicked.connect(self._flip)
        for button in (self.crop_button, self.flip_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            buttons.addWidget(button)
        layout.addLayout(row)
        layout.addLayout(buttons)
        return frame

    def _build_settings_card(self) -> QFrame:
        frame, layout = card(spacing=8)
        layout.addWidget(caption("Réglages"))
        layout.addWidget(QLabel("Moteur d'analyse"))
        self.engine_choice = Segmented([("Stockfish", None), ("Maison (Python)", HOMEMADE)])
        self.engine_choice.setToolTip(
            "Stockfish : le plus fort. Maison : le moteur du projet (lot L3), bien plus faible."
        )
        self.engine_choice.set_value(self._settings.value("engine", None) or None)
        self.engine_choice.changed.connect(self._on_engine_changed)
        layout.addWidget(self.engine_choice)
        layout.addSpacing(6)
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Temps de réflexion"))
        time_row.addStretch(1)
        self.time_value = QLabel(seconds_text(DEFAULT_TIME))
        self.time_value.setStyleSheet("font-weight: 700;")
        time_row.addWidget(self.time_value)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(1, 100)  # dixièmes de seconde : 0,1 à 10 s (ENF-05)
        self.time_slider.setValue(int(DEFAULT_TIME * 10))
        self.time_slider.valueChanged.connect(
            lambda value: self.time_value.setText(seconds_text(value / 10))
        )
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(450)
        self._settings_timer.timeout.connect(self._on_settings_changed)
        self.time_slider.valueChanged.connect(lambda _: self._settings_timer.start())
        self.multipv = Segmented([("1 coup", 1), ("2 coups", 2), ("3 coups", 3)])
        self.multipv.set_value(3)
        self.multipv.changed.connect(lambda _: self._on_settings_changed())
        self.orientation = Segmented(
            [
                ("Auto", Orientation.AUTO),
                ("Blancs bas", Orientation.WHITE_BOTTOM),
                ("Noirs bas", Orientation.BLACK_BOTTOM),
            ]
        )
        self.orientation.setToolTip("Auto : d'après les coordonnées écrites sur l'échiquier")
        self.orientation.changed.connect(lambda _: self._reread())
        layout.addLayout(time_row)
        layout.addWidget(self.time_slider)
        layout.addSpacing(6)
        layout.addWidget(QLabel("Coups proposés"))
        layout.addWidget(self.multipv)
        layout.addSpacing(6)
        layout.addWidget(QLabel("Orientation de l'échiquier"))
        layout.addWidget(self.orientation)
        return frame

    def _build_actions(self) -> None:
        menu = QMenu(self)
        entries = [
            ("open", "Ouvrir une capture…", QKeySequence.StandardKey.Open, self._open_file),
            ("paste", "Coller une capture", QKeySequence.StandardKey.Paste, self._paste),
            None,
            ("play", "Jouer contre l'IA…", QKeySequence("Ctrl+J"), self._open_play),
            ("target", "Mode capture (téléphone ou écran)…", None, self._offer_capture_mode),
            ("crop", "Délimiter le plateau à la main…", None, self._draw_region),
            ("flip", "Retourner l'échiquier", QKeySequence("F"), self._flip),
            ("target", "Calibrer sur la position initiale (facultatif)…", None, self._recalibrate),
            None,
            ("copy", "Copier la FEN", QKeySequence("Ctrl+Shift+C"), self._copy_fen),
            ("copy", "Copier la variante (PGN)", None, self._copy_pgn),
            None,
            ("deeper", "Rechercher une mise à jour…", None, lambda: self.updater.check()),
            ("copy", "Partager Chessio…", None, lambda: share(self)),
            ("info", "Cadre d'utilisation", None, self._show_warning),
            ("close", "Quitter", QKeySequence.StandardKey.Quit, self.close),
        ]
        for entry in entries:
            if entry is None:
                menu.addSeparator()
                continue
            name, text, shortcut, slot = entry
            action = QAction(icon(name), text, self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            menu.addAction(action)
            self.addAction(action)  # raccourci actif même menu fermé
        auto = QAction("Vérifier les mises à jour au démarrage", self)
        auto.setCheckable(True)
        auto.setChecked(Updater.check_on_start())
        auto.toggled.connect(Updater.set_check_on_start)
        menu.insertAction(menu.actions()[-3], auto)
        self.menu_button.setMenu(menu)
        for key, slot in (
            ("W", lambda: self._on_turn_clicked(chess.WHITE, from_key=True)),
            ("B", lambda: self._on_turn_clicked(chess.BLACK, from_key=True)),
            ("Space", lambda: self._start_analysis(UNLIMITED)),
            ("Escape", self._stop_analysis),
            ("F5", self._capture),
        ):
            QShortcut(QKeySequence(key), self, activated=slot)

    def _start_worker(self) -> None:
        self.worker_thread = QThread(self)
        self.worker = Worker(self.engine_choice.value())
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.start_engine)
        self.request_recognize.connect(self.worker.recognize)
        self.request_analyse.connect(self.worker.analyse)
        self.request_calibrate.connect(self.worker.calibrate)
        self.request_phone.connect(self.worker.capture_phone)
        self.worker.phone_captured.connect(self._on_phone_captured)
        self.request_engine.connect(self.worker.switch_engine)
        self.request_stop.connect(self.worker.stop, Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.engine_ready.connect(self._on_engine_ready)
        self.worker.engine_failed.connect(self._on_engine_failed)
        self.worker.recognized.connect(self._on_recognized)
        self.worker.progress.connect(self._on_progress)
        self.worker.analysed.connect(self._on_analysed)
        self.worker.calibrated.connect(self._on_calibrated)
        self.worker.failed.connect(self._on_failed)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(50)
        self._progress_timer.timeout.connect(self._tick_progress)
        self.worker_thread.start()

    # --- état général ---------------------------------------------------------

    def _load_templates(self) -> TemplateSet | None:
        path = templates_path()
        try:
            return TemplateSet.load(path) if path.exists() else None
        except (OSError, ValueError, KeyError):
            return None

    @staticmethod
    def _has_model() -> bool:
        """Réseau des cases disponible : aucune calibration n'est alors nécessaire."""
        return load_model() is not None

    def _next_generation(self) -> int:
        self.generation += 1
        self.worker.latest = self.generation
        return self.generation

    def _show_empty(self) -> None:
        self.panel.hint.set_calibrated(self.templates is not None or self._has_model())
        self.panel.show_hint(True)
        self.banner.hide()
        self._clear_analysis("En attente d'une capture.")
        self._update_turn_hint()
        self._update_capture_card()

    def _on_engine_ready(self, name: str) -> None:
        self.engine_ready = True
        self.engine_chip.set_full_text(f"●  {name}")
        self.engine_chip.setStyleSheet(f"color: {PALETTE['accent']}; background: transparent;")
        if self._pending_analysis:
            self._pending_analysis = False
            self._refresh()

    def _on_engine_changed(self, choice: str | None) -> None:
        self._settings.setValue("engine", choice or "")
        self._next_generation()  # abandonne l'analyse en cours
        self.engine_ready = False
        self.engine_chip.set_full_text("●  Moteur : démarrage…")
        self.engine_chip.setStyleSheet(f"color: {PALETTE['muted']}; background: transparent;")
        if self.board is not None:
            self._pending_analysis = True
            self._clear_analysis("Changement de moteur…")
        self.request_engine.emit(choice)

    def _on_engine_failed(self, message: str) -> None:
        self.engine_chip.set_full_text("●  Moteur indisponible")
        self.engine_chip.setStyleSheet(f"color: {PALETTE['danger']}; background: transparent;")
        self.banner.show_message("danger", f"Moteur d'échecs indisponible : {message}")

    def _toast(self, text: str) -> None:
        self.toast.show_message(text)

    # --- arrivée d'une capture (EF-01) ---------------------------------------

    def _paste(self) -> None:
        clipboard = QGuiApplication.clipboard()
        mime = clipboard.mimeData()
        if mime.hasImage() and not clipboard.image().isNull():
            self._set_image(qimage_to_rgb(clipboard.image()))
        elif mime.hasUrls() and any(url.isLocalFile() for url in mime.urls()):
            path = next(url.toLocalFile() for url in mime.urls() if url.isLocalFile())
            self._set_image_file(path)
        elif mime.hasText() and mime.text().strip():
            self.fen_edit.setText(mime.text().strip())
            self._load_fen()
        else:
            self._toast("Le presse-papiers ne contient ni image ni FEN.")

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir une capture", "", IMAGE_FILTER)
        if path:
            self._set_image_file(path)

    def _set_image_file(self, path: str) -> None:
        try:
            self._set_image(load_image(path))
        except OSError as exc:
            self._toast(f"Image illisible : {exc}")

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasImage() or event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        data = event.mimeData()
        if data.hasImage():
            self._set_image(qimage_to_rgb(QImage(data.imageData())))
        elif data.hasUrls():
            local = [url.toLocalFile() for url in data.urls() if url.isLocalFile()]
            if local:
                self._set_image_file(local[0])

    def _set_image(self, rgb: np.ndarray) -> None:
        self.image = rgb
        self.region = None
        self.turn_explicit = False
        if self.my_color is not None:
            self.turn.set_value(self.my_color)  # à défaut d'indice contraire : à vous de jouer
        self.orientation.set_value(Orientation.AUTO)
        self._update_capture_card()
        if self.templates is None and not self._has_model():
            self._calibrate_from(rgb)  # sans modèle entraîné, la calibration reste nécessaire
        else:
            self._reread()

    def _draw_region(self) -> None:
        if self.image is None:
            self._toast("Collez d'abord une capture.")
            return
        dialog = RegionDialog(self.image, self)
        if dialog.exec() and dialog.region is not None:
            self.region = dialog.region
            self._reread()

    # --- mode capture (parties contre les bots) ----------------------------

    def _open_play(self) -> None:
        from chessio.ui.play import PlayWindow

        self.play_window = PlayWindow()
        self.play_window.resize(1280, 840)
        self.play_window.show()

    def _offer_capture_mode(self) -> None:
        """Propose de capturer le téléphone branché, à défaut l'écran de l'ordinateur."""
        found = phones()
        ready = [phone for phone in found if phone.ready]
        if ready:
            phone = ready[0]
            answer = QMessageBox.question(
                self,
                "Téléphone connecté",
                f"Un téléphone est connecté ({phone.model or phone.serial}).\n\n"
                "Jouez-vous aux échecs dessus, contre un bot ? Chessio capturera son écran "
                "à chaque clic sur « Capturer » (F5).",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.capture_source, self.phone_serial = "phone", phone.serial
                self._ask_color()
            return
        if found:
            QMessageBox.information(
                self,
                "Téléphone à autoriser",
                "Un téléphone est branché mais n'autorise pas encore Chessio. Sur le "
                "téléphone, activez Options pour les développeurs → Débogage USB, puis "
                "acceptez la demande d'autorisation qui s'affiche.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Capture de l'écran",
            "Aucun téléphone n'est connecté.\n\nJouez-vous aux échecs sur cet ordinateur, "
            "contre un bot ? Vous indiquerez une fois la zone de l'écran où se trouve "
            "l'échiquier ; chaque clic sur « Capturer » (F5) la lira.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.capture_source, self.screen_zone = "screen", None
            self._ask_color()
            self._capture()

    def _ask_color(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Votre camp")
        box.setText(
            "Avec quelles pièces jouez-vous ? Chessio calculera les coups de ce camp.\n\n"
            "Rappel : uniquement contre les bots (règles de Chess.com)."
        )
        white = box.addButton("○  Blancs", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("●  Noirs", QMessageBox.ButtonRole.AcceptRole)
        box.exec()
        self.my_color = chess.WHITE if box.clickedButton() is white else chess.BLACK
        self.turn.set_value(self.my_color)
        side = "Blancs" if self.my_color == chess.WHITE else "Noirs"
        self._toast(f"Mode capture : vous jouez les {side}. Appuyez sur F5 à votre tour.")

    def _capture(self) -> None:
        """Une capture, déclenchée par l'utilisateur."""
        if self.capture_source is None:
            self._offer_capture_mode()
            return
        generation = self._next_generation()  # interrompt une analyse en cours
        self.status.setText("Capture…")
        if self.capture_source == "phone":
            self.request_phone.emit(generation, self.phone_serial)
        else:
            self.screen.request()

    def _on_phone_captured(self, generation: int, rgb: np.ndarray) -> None:
        if generation == self.generation:
            self._set_image(rgb)

    def _on_screen_captured(self, rgb: np.ndarray) -> None:
        if self.screen_zone is None:
            dialog = RegionDialog(rgb, self)
            dialog.setWindowTitle("Zone de l'échiquier à capturer")
            if not dialog.exec() or dialog.region is None:
                self.status.setText("")
                return
            self.screen_zone = dialog.region
        zone = self.screen_zone
        crop = rgb[zone.y : zone.y + zone.size, zone.x : zone.x + zone.size]
        self._set_image(np.ascontiguousarray(crop) if crop.size else rgb)

    # --- calibration ----------------------------------------------------------

    def _calibrate_from(self, rgb: np.ndarray) -> None:
        self.calibrating = True
        generation = self._next_generation()
        self.status.setText("Calibration sur la position initiale…")
        self.request_calibrate.emit(generation, rgb)

    def _recalibrate(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Capture de la position initiale (votre thème)", "", IMAGE_FILTER
        )
        if path:
            try:
                self.image = load_image(path)
            except OSError as exc:
                self._toast(f"Image illisible : {exc}")
                return
            self._update_capture_card()
            self._calibrate_from(self.image)

    def _on_calibrated(self, templates: TemplateSet) -> None:
        self.calibrating = False
        self.templates = templates
        self.panel.hint.set_calibrated(True)
        self._toast("Calibration réussie : vos pièces et vos couleurs sont enregistrées.")
        if self.image is not None:
            self._reread()

    # --- lecture de la capture ------------------------------------------------

    def _reread(self) -> None:
        """Relit la capture (nouvelle image, orientation ou zone imposée)."""
        if self.image is None or (self.templates is None and not self._has_model()):
            return
        generation = self._next_generation()
        self._stop_progress()
        self.status.setText("Lecture de la capture…")
        turn = self.turn.value() if self.turn_explicit else None
        self.request_recognize.emit(
            generation,
            self.image,
            turn,
            self.turn.value(),
            self.templates,
            self.orientation.value(),
            self.region,
        )

    def _on_recognized(self, generation: int, result: Recognition) -> None:
        if generation != self.generation:
            return
        self.recognition = result
        self.board = result.board.copy()
        self.white_bottom = result.white_bottom
        self.lastmove = result.last_move
        self.doubtful = set(result.doubtful)
        self.verified = set()
        self.turn_source = result.turn_source
        self.turn.set_value(result.board.turn)
        self.panel.show_hint(False)
        bottom = "Blancs" if result.white_bottom else "Noirs"
        self.status.setText(
            f"Capture lue en {result.elapsed * 1000:.0f} ms · {bottom} en bas "
            f"(d'après : {result.orientation_source})"
        )
        self._update_turn_hint()
        self._update_capture_card()
        self._refresh()

    def _on_failed(self, generation: int, message: str) -> None:
        if generation != self.generation:
            return
        self._stop_progress()
        if self.calibrating:
            self.calibrating = False
            self.panel.show_hint(True)
            self.banner.show_message(
                "danger",
                f"{message}\nPour la première utilisation, collez une capture de la position "
                "initiale (Blancs en bas, dans votre thème habituel).",
            )
            self.status.setText("")
            return
        if self.board is None:
            self.panel.show_hint(True)
            crop = QPushButton(icon("crop"), "  Délimiter…")
            crop.clicked.connect(self._draw_region)
            self.banner.show_message("danger", message, [crop])
        else:
            self.banner.show_message("danger", message)
        self.status.setText("")
        self._update_capture_card(error=message)

    # --- position, vérifications et corrections (EF-07) -----------------------

    def _refresh(self) -> None:
        """Affiche la position courante ; lance l'analyse si rien ne s'y oppose."""
        board = self.board
        if board is None:
            return
        pending = sorted(self.doubtful - self.verified)
        rules = validate(board)
        marked = set(pending) | {sq for issue in rules for sq in issue.squares}
        self.panel.board.interactive = True
        self.panel.board.set_position(
            board, white_bottom=self.white_bottom, marked=marked, lastmove=self.lastmove
        )
        self.panel.eval_bar.set_eval(None, "", self.white_bottom)
        self.fen_edit.setText(board.fen())
        self._update_capture_card()

        if self.recognition is not None and self.recognition.orientation_assumed:
            white = QPushButton("Blancs en bas")
            white.clicked.connect(lambda: self._set_orientation(Orientation.WHITE_BOTTOM))
            black = QPushButton("Noirs en bas")
            black.clicked.connect(lambda: self._set_orientation(Orientation.BLACK_BOTTOM))
            self.banner.show_message(
                "warning",
                "Orientation non lue (coordonnées invisibles sur la capture). "
                "Quel camp est en bas de l'échiquier ?",
                [white, black],
            )
            self._clear_analysis("Indiquez l'orientation pour obtenir un coup.")
            return
        if pending:
            accept = QPushButton(icon("check"), "  Tout est correct")
            accept.clicked.connect(self._accept_doubtful)
            names = ", ".join(chess.square_name(sq) for sq in pending)
            self.banner.show_message(
                "warning",
                f"Case{'s' if len(pending) > 1 else ''} incertaine{'s' if len(pending) > 1 else ''}"
                f" ({names}), en rouge : cliquez dessus pour choisir la bonne pièce.",
                [accept],
            )
            self._clear_analysis("Vérifiez la position pour obtenir un coup.")
            return
        if rules:
            self.banner.show_message(
                "danger",
                "Position impossible : "
                + " ; ".join(str(issue) for issue in rules)
                + ". Cliquez sur une case pour la corriger.",
            )
            self._clear_analysis("Corrigez la position pour obtenir un coup.")
            return

        detected = self.recognition is not None and self.recognition.turn_detected
        if self.my_color is not None and detected and board.turn != self.my_color:
            again = QPushButton(icon("target"), "  Capturer à nouveau")
            again.clicked.connect(self._capture)
            self.banner.show_message(
                "warning",
                "C'est au bot de jouer. Capturez à nouveau (F5) après son coup.",
                [again],
            )
            self._clear_analysis("En attente du coup du bot.")
            return
        if detected and self.turn_explicit and board.turn != self.recognition.board.turn:
            self.banner.show_message(
                "info",
                f"La capture indique plutôt le trait aux "
                f"{'Blancs' if self.recognition.board.turn else 'Noirs'} "
                f"({self.recognition.turn_source}).",
            )
        else:
            self.banner.hide()
        self._start_analysis()

    def _set_orientation(self, orientation: Orientation) -> None:
        self.orientation.set_value(orientation)
        self._reread()

    def _accept_doubtful(self) -> None:
        self.verified |= self.doubtful
        self._refresh()

    def _on_square_clicked(self, square: chess.Square, position: QPoint) -> None:
        if self.board is None:
            return
        menu = QMenu(self)
        title = menu.addAction(f"Case {chess.square_name(square)}")
        title.setEnabled(False)
        menu.addSeparator()
        for symbols in ("KQRBNP", "kqrbnp"):
            for symbol in symbols:
                action = menu.addAction(piece_icon(symbol), PIECE_NAMES[symbol])
                action.setData(symbol)
            menu.addSeparator()
        empty = menu.addAction(icon("close"), "Case vide")
        empty.setData("")
        chosen = menu.exec(position)
        if chosen is not None and chosen.data() is not None:
            self._correct_square(square, chosen.data())

    def _correct_square(self, square: chess.Square, symbol: str) -> None:
        """Remplace la pièce d'une case (« » : case vide) ; la case est alors vérifiée."""
        assert self.board is not None
        if symbol:
            self.board.set_piece_at(square, chess.Piece.from_symbol(symbol))
        else:
            self.board.remove_piece_at(square)
        self.board.castling_rights = self.board.clean_castling_rights()
        self.board.ep_square = None
        self.verified.add(square)
        self.lastmove = None
        if not self.turn_explicit and self.turn_source in (TURN_CHECK, TURN_DEFAULT):
            checked = turn_from_check(self.board)
            if checked is not None:
                self.board.turn, self.turn_source = checked, TURN_CHECK
                self.turn.set_value(checked)
            else:
                self.turn_source = TURN_DEFAULT
            self._update_turn_hint()
        self._refresh()

    def _load_fen(self) -> None:
        text = self.fen_edit.text().strip()
        try:
            board = chess.Board(text)
        except ValueError:
            self._toast("FEN invalide.")
            return
        self.board = board
        self.recognition = None
        self.doubtful.clear()
        self.verified.clear()
        self.lastmove = None
        self.turn_explicit = True
        self.turn_source = TURN_INPUT
        self.turn.set_value(board.turn)
        self.panel.show_hint(False)
        self.status.setText("Position saisie à la main.")
        self._update_turn_hint()
        self._refresh()

    def _flip(self) -> None:
        if self.image is not None and self.templates is not None and self.recognition is not None:
            self._set_orientation(
                Orientation.BLACK_BOTTOM if self.white_bottom else Orientation.WHITE_BOTTOM
            )
        elif self.board is not None:
            self.white_bottom = not self.white_bottom
            self._redraw_board()

    # --- trait (EF-03) -----------------------------------------------------------

    def _on_turn_clicked(self, color: chess.Color, from_key: bool = False) -> None:
        if from_key:
            self.turn.set_value(color)
        self.turn_explicit = True
        self.turn_source = TURN_INPUT
        self._update_turn_hint()
        if self.board is None:
            return
        if self.recognition is not None and self.recognition.orientation_assumed:
            self._reread()
            return
        if self.board.turn != color:
            self.board.turn = color
            self.board.ep_square = None
        self._refresh()

    def _update_turn_hint(self) -> None:
        warn = False
        if self.board is None:
            text = "Détecté automatiquement d'après le dernier coup joué. Touches W / B."
        elif self.turn_source == TURN_INPUT:
            text = "Choisi par vous (touches W / B)."
        elif self.turn_source == TURN_LAST_MOVE:
            move = self.recognition.last_move if self.recognition is not None else None
            squares = (
                f" ({chess.square_name(move.from_square)} → {chess.square_name(move.to_square)})"
                if move is not None
                else ""
            )
            text = f"Détecté d'après le dernier coup joué{squares}."
        elif self.turn_source == TURN_CHECK:
            text = "Détecté : ce roi est en échec."
        else:
            text = "Non détecté (aucun dernier coup surligné) : vérifiez ce choix !"
            warn = True
        self.turn_hint.setText(text)
        self.turn_hint.setObjectName("HintWarning" if warn else "Hint")
        self.turn_hint.style().unpolish(self.turn_hint)
        self.turn_hint.style().polish(self.turn_hint)

    # --- analyse (EF-08, EF-11) ----------------------------------------------

    def _seconds(self) -> float:
        return self.time_slider.value() / 10

    def _on_settings_changed(self) -> None:
        if self.board is not None and self.analysis is not None:
            self._refresh()

    def _start_analysis(self, seconds: float | None = None) -> None:
        if self.board is None or self.banner.isVisible() and self._blocking_banner():
            return
        if not self.engine_ready:
            self._pending_analysis = True
            self._clear_analysis("Démarrage du moteur…")
            return
        seconds = self._seconds() if seconds is None else seconds
        generation = self._next_generation()
        self.analysis = None
        self.selected_line = 0
        self._analysis_started = time.monotonic()
        self._analysis_seconds = seconds
        self.best_label.setText("…")
        self.eval_chip.hide()
        unlimited = seconds == UNLIMITED
        self.best_detail.setText(
            "Analyse illimitée : le moteur approfondit… (Échap pour arrêter)"
            if unlimited
            else f"Analyse ({seconds_text(seconds)})…"
        )
        self.progress.setRange(0, 0 if unlimited else 1000)  # barre animée sans fin
        self.progress.setValue(0)
        self.progress.show()
        self.stop_button.show()
        self.deeper_button.hide()
        self._progress_timer.start()
        self.request_analyse.emit(generation, self.board.fen(), seconds, int(self.multipv.value()))

    def _blocking_banner(self) -> bool:
        return self.banner.property("kind") in ("warning", "danger")

    def _stop_analysis(self) -> None:
        if not self._progress_timer.isActive():
            return
        self._next_generation()  # interrompt l'analyse en cours
        self._stop_progress()
        if self.analysis is not None:
            self._show_analysis(self.analysis, final=True)

    def _tick_progress(self) -> None:
        if self._analysis_seconds == UNLIMITED:
            return
        elapsed = time.monotonic() - self._analysis_started
        self.progress.setValue(min(1000, int(1000 * elapsed / self._analysis_seconds)))

    def _stop_progress(self) -> None:
        self._progress_timer.stop()
        self.progress.hide()
        self.stop_button.hide()
        self.deeper_button.show()

    def _on_progress(self, generation: int, analysis: Analysis) -> None:
        if generation == self.generation:
            self._show_analysis(analysis, final=False)

    def _on_analysed(self, generation: int, analysis: Analysis) -> None:
        if generation == self.generation:
            self._stop_progress()
            self._show_analysis(analysis, final=True)

    def _on_line_clicked(self, index: int) -> None:
        if self.analysis is not None and index < len(self.analysis.lines):
            self.selected_line = index
            self._show_analysis(self.analysis, final=not self._progress_timer.isActive())

    def _show_analysis(self, analysis: Analysis, *, final: bool) -> None:
        self.analysis = analysis
        if analysis.outcome:
            mated = analysis.outcome == "mat"
            self.best_label.setText("Échec et mat" if mated else "Pat")
            self.best_detail.setText("La partie est terminée.")
            self.eval_chip.hide()
            loser = chess.Board(analysis.fen).turn
            expectation = (0.0 if loser == chess.WHITE else 1.0) if mated else 0.5
            self.panel.eval_bar.set_eval(expectation, "#" if mated else "½", self.white_bottom)
            self.wdl_bar.set_wdl(None)
            self.wdl_label.setText("")
            self._show_lines(())
            return
        lines = analysis.lines
        if not lines:
            return
        best = lines[0]
        self.best_label.setText(figurine(best.san, self.board.turn))
        self.eval_chip.setText(best.evaluation)
        self.eval_chip.show()
        wdl = best.white_wdl
        self.wdl_bar.set_wdl(wdl)
        self.wdl_label.setText(
            f"Blancs {wdl.wins / 10:.0f} %   ·   Nulle {wdl.draws / 10:.0f} %   ·   "
            f"Noirs {wdl.losses / 10:.0f} %"
        )
        state = "" if final else "  (en cours)"
        self.best_detail.setText(
            f"{self._describe(best.move)} · profondeur {best.depth} · "
            f"{seconds_text(analysis.elapsed)}{state}"
        )
        mate = best.score.white().mate()
        short = f"M{abs(mate)}" if mate is not None else best.evaluation.lstrip("+")
        self.panel.eval_bar.set_eval(best.white_expectation, short, self.white_bottom)
        self._show_lines(lines)
        selected = min(self.selected_line, len(lines) - 1)
        order = [i for i in range(len(lines)) if i != selected] + [selected]
        self.panel.board.set_position(
            self.board,
            white_bottom=self.white_bottom,
            arrows=[(lines[i].move, ARROW_NAMES[i]) for i in order],
            lastmove=self.lastmove,
        )

    def _describe(self, move: chess.Move) -> str:
        assert self.board is not None
        piece = self.board.piece_at(move.from_square)
        name = PIECE_NAMES[piece.symbol()].split()[0] if piece else "Pièce"
        return f"{name} {chess.square_name(move.from_square)} → {chess.square_name(move.to_square)}"

    def _show_lines(self, lines) -> None:
        selected = min(self.selected_line, max(0, len(lines) - 1))
        for index, row in enumerate(self.line_rows):
            if index < len(lines):
                line = lines[index]
                after = self.board.copy(stack=False)
                after.push(line.move)
                row.set_line(
                    figurine(line.san, self.board.turn),
                    figurine_line(after, line.pv[1:]),
                    line.evaluation,
                    LINE_COLORS[index],
                    index == selected,
                )
                row.show()
            else:
                row.hide()
        self.lines_empty.setVisible(not lines)

    def _clear_analysis(self, message: str) -> None:
        self._stop_progress()
        self.analysis = None
        self.best_label.setText("—")
        self.eval_chip.hide()
        self.best_detail.setText(message)
        self.wdl_bar.set_wdl(None)
        self.wdl_label.setText("")
        self._show_lines(())

    def _redraw_board(self) -> None:
        if self.board is None:
            return
        if self.analysis is not None:
            self._show_analysis(self.analysis, final=not self._progress_timer.isActive())
        else:
            self.panel.board.set_position(
                self.board, white_bottom=self.white_bottom, lastmove=self.lastmove
            )

    # --- carte « capture » --------------------------------------------------------

    def _update_capture_card(self, error: str | None = None) -> None:
        has_image = self.image is not None
        self.crop_button.setEnabled(has_image)
        self.flip_button.setEnabled(self.board is not None)
        if not has_image:
            self.thumbnail.setPixmap(icon("image", PALETTE["faint"], 36).pixmap(36, 36))
            self.capture_status.setText("Aucune capture")
            self.capture_detail.setText("Ctrl+V, glisser-déposer ou Ouvrir.")
            return
        pixmap = rgb_to_pixmap(self.image).scaled(
            self.thumbnail.size() - QSize(8, 8),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.thumbnail.setPixmap(pixmap)
        if error:
            self.capture_status.setText("Lecture impossible")
            self.capture_detail.setText("Essayez « Délimiter… » pour indiquer le plateau.")
        elif self.calibrating:
            self.capture_status.setText("Calibration…")
            self.capture_detail.setText("Apprentissage de vos pièces et couleurs.")
        elif self.recognition is None:
            self.capture_status.setText("Capture chargée")
            self.capture_detail.setText("")
        else:
            pending = len(self.doubtful - self.verified)
            self.capture_status.setText(
                f"{pending} case{'s' if pending > 1 else ''} à vérifier"
                if pending
                else "Position lue"
            )
            size = self.recognition.region.size
            self.capture_detail.setText(
                f"Plateau de {size} px, lu en {self.recognition.elapsed * 1000:.0f} ms."
            )

    # --- presse-papiers et divers ---------------------------------------------

    def _copy_fen(self) -> None:
        if self.board is None:
            self._toast("Aucune position à copier.")
            return
        QGuiApplication.clipboard().setText(self.board.fen())
        self._toast("FEN copiée dans le presse-papiers.")

    def _copy_pgn(self) -> None:
        if self.board is None or self.analysis is None or not self.analysis.lines:
            self._toast("Aucune analyse à copier.")
            return
        line = self.analysis.lines[min(self.selected_line, len(self.analysis.lines) - 1)]
        game = chess.pgn.Game()
        game.headers["Event"] = "Analyse Chessio"
        game.setup(self.board)
        node = game
        for move in line.pv:
            node = node.add_variation(move)
        node.comment = f"Évaluation {line.evaluation}, profondeur {line.depth}"
        QGuiApplication.clipboard().setText(str(game))
        self._toast("Variante copiée au format PGN.")

    def _show_warning(self) -> None:
        QMessageBox.information(self, "Cadre d'utilisation", WARNING)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.toast.isVisible():
            self.toast.show_message(self.toast.text())

    def closeEvent(self, event) -> None:  # noqa: N802
        self._next_generation()
        self.request_stop.emit()
        self.worker_thread.quit()
        self.worker_thread.wait(3000)
        super().closeEvent(event)


__all__ = ["MainWindow", "WARNING", "__version__"]
