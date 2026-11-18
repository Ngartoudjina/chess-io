"""Composants visuels de l'interface."""

from __future__ import annotations

from collections.abc import Iterable

import chess
import chess.engine
import chess.svg
from PySide6.QtCore import (
    QByteArray,
    QEasingCurve,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from chessio.ui.style import (
    BOARD_COLORS,
    COORD_ON_DARK,
    COORD_ON_LIGHT,
    DOUBTFUL_FILL,
    PALETTE,
    icon,
)
from chessio.vision.board import screen_to_square


def card(*, spacing: int = 10, margins: int = 16) -> tuple[QFrame, QVBoxLayout]:
    """Carte arrondie et sa mise en page verticale."""
    frame = QFrame()
    frame.setObjectName("Card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(margins, margins, margins, margins)
    layout.setSpacing(spacing)
    return frame, layout


def fit_to_screen(window: QWidget, width: int, height: int) -> None:
    """Taille d'ouverture limitée à l'écran (barre du haut et barre de titre comprises).

    Sous Wayland, la zone « disponible » inclut souvent la barre de GNOME : on garde
    une marge pour que le bas de la fenêtre (ligne d'état, champ FEN) reste visible.
    """
    screen = window.screen() or QGuiApplication.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        width = min(width, area.width() - 40)
        height = min(height, area.height() - 90)
    window.resize(max(width, window.minimumWidth()), max(height, window.minimumHeight()))


def caption(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("Caption")
    return label


def restyle(widget: QWidget) -> None:
    """Réapplique la feuille de style après changement d'une propriété dynamique."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class BoardView(QWidget):
    """Échiquier vectoriel : position, flèches, cases à vérifier ; clic sur une case."""

    square_clicked = Signal(int, QPoint)  # case, position globale du clic

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._renderer = QSvgRenderer(self)
        self._board = chess.Board()
        self._white_bottom = True
        self._hover: chess.Square | None = None
        self.interactive = False
        self.set_position(chess.Board())

    @property
    def white_bottom(self) -> bool:
        return self._white_bottom

    def set_position(
        self,
        board: chess.Board,
        *,
        white_bottom: bool = True,
        arrows: Iterable[tuple[chess.Move, str]] = (),
        marked: Iterable[chess.Square] = (),
        lastmove: chess.Move | None = None,
        fill: dict[chess.Square, str] | None = None,
    ) -> None:
        """``fill`` : couleurs supplémentaires de cases (sélection, cibles possibles)."""
        self._board = board.copy(stack=False)
        self._white_bottom = white_bottom
        check = None
        if self._board.is_valid() and self._board.is_check():
            check = self._board.king(self._board.turn)
        # Flèches dessinées dans l'ordre donné : la dernière passe par-dessus.
        arrow_list = [
            chess.svg.Arrow(move.from_square, move.to_square, color=color) for move, color in arrows
        ]
        svg = chess.svg.board(
            self._board,
            orientation=chess.WHITE if white_bottom else chess.BLACK,
            lastmove=lastmove,
            check=check,
            arrows=arrow_list,
            fill={**(fill or {}), **dict.fromkeys(marked, DOUBTFUL_FILL)},
            coordinates=False,
            colors=BOARD_COLORS,
        )
        self._renderer.load(QByteArray(svg.encode()))
        self.update()

    def board_rect(self) -> QRectF:
        side = min(self.width(), self.height())
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def paintEvent(self, event) -> None:  # noqa: N802 (API Qt)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.board_rect()
        clip = QPainterPath()
        clip.addRoundedRect(rect, 8, 8)
        painter.setClipPath(clip)
        self._renderer.render(painter, rect)
        self._paint_coordinates(painter, rect)
        if self.interactive and self._hover is not None:
            row, col = self._screen_cell(self._hover)
            size = rect.width() / 8
            painter.setPen(QPen(QColor(255, 255, 255, 170), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRectF(rect.x() + col * size, rect.y() + row * size, size, size).adjusted(
                    1.5, 1.5, -1.5, -1.5
                )
            )

    def _paint_coordinates(self, painter: QPainter, rect: QRectF) -> None:
        """Coordonnées dans les cases, comme sur Chess.com."""
        size = rect.width() / 8
        font = QFont(self.font())
        font.setBold(True)
        font.setPixelSize(max(8, int(size * 0.17)))
        painter.setFont(font)
        pad = size * 0.06
        for index in range(8):
            # Chiffre en haut à gauche de la colonne de gauche.
            square = screen_to_square(index, 0, self._white_bottom)
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            painter.setPen(QColor(COORD_ON_LIGHT if light else COORD_ON_DARK))
            cell = QRectF(rect.x(), rect.y() + index * size, size, size)
            painter.drawText(
                cell.adjusted(pad, pad * 0.5, 0, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                str(chess.square_rank(square) + 1),
            )
            # Lettre en bas à droite de la rangée du bas.
            square = screen_to_square(7, index, self._white_bottom)
            light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
            painter.setPen(QColor(COORD_ON_LIGHT if light else COORD_ON_DARK))
            cell = QRectF(rect.x() + index * size, rect.y() + 7 * size, size, size)
            painter.drawText(
                cell.adjusted(0, 0, -pad, -pad * 0.4),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                chess.FILE_NAMES[chess.square_file(square)],
            )

    def _screen_cell(self, square: chess.Square) -> tuple[int, int]:
        file, rank = chess.square_file(square), chess.square_rank(square)
        return (7 - rank, file) if self._white_bottom else (rank, 7 - file)

    def _square_at(self, pos: QPointF) -> chess.Square | None:
        rect = self.board_rect()
        if not rect.contains(pos):
            return None
        size = rect.width() / 8
        col = min(7, int((pos.x() - rect.x()) / size))
        row = min(7, int((pos.y() - rect.y()) / size))
        return screen_to_square(row, col, self._white_bottom)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        square = self._square_at(event.position()) if self.interactive else None
        if square != self._hover:
            self._hover = square
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if square is not None
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = None
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self.interactive or event.button() != Qt.MouseButton.LeftButton:
            return
        square = self._square_at(event.position())
        if square is not None:
            self.square_clicked.emit(square, event.globalPosition().toPoint())


class EvalBar(QWidget):
    """Barre d'évaluation verticale : part des Blancs selon leurs chances (animée)."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(26)
        self._shown = 0.5
        self._text = ""
        self._active = False
        self._white_bottom = True
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(450)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._on_value)

    def set_eval(self, expectation: float | None, text: str, white_bottom: bool) -> None:
        self._text = text
        self._white_bottom = white_bottom
        self._active = expectation is not None
        target = 0.5 if expectation is None else min(0.98, max(0.02, expectation))
        self._animation.stop()
        self._animation.setStartValue(self._shown)
        self._animation.setEndValue(target)
        self._animation.start()
        self.update()

    def _on_value(self, value: float) -> None:
        self._shown = float(value)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(rect, 6, 6)
        painter.setClipPath(clip)
        painter.fillRect(rect, QColor("#2a2825"))
        if not self._active:
            painter.fillRect(rect, QColor(PALETTE["card_alt"]))  # neutre : pas d'analyse
            return
        white_height = rect.height() * self._shown
        if self._white_bottom:
            white = QRectF(rect.x(), rect.bottom() - white_height, rect.width(), white_height)
        else:
            white = QRectF(rect.x(), rect.y(), rect.width(), white_height)
        painter.fillRect(white, QColor(PALETTE["white_piece"]))
        if self._text:
            font = QFont(self.font())
            font.setPixelSize(10)
            font.setBold(True)
            painter.setFont(font)
            white_leads = self._shown >= 0.5
            # Le chiffre s'écrit du côté du camp qui mène, dans sa couleur.
            at_bottom = white_leads == self._white_bottom
            painter.setPen(QColor("#1a1917" if white_leads else PALETTE["text"]))
            text_rect = QRectF(rect.x(), rect.y() + 4, rect.width(), rect.height() - 8)
            align = Qt.AlignmentFlag.AlignHCenter | (
                Qt.AlignmentFlag.AlignBottom if at_bottom else Qt.AlignmentFlag.AlignTop
            )
            painter.drawText(text_rect, align, self._text)


class WdlBar(QWidget):
    """Chances Blancs / nulle / Noirs en une barre horizontale."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(10)
        self._wdl: tuple[int, int, int] | None = None

    def set_wdl(self, wdl: chess.engine.Wdl | None) -> None:
        self._wdl = None if wdl is None else (wdl.wins, wdl.draws, wdl.losses)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(rect, 5, 5)
        painter.setClipPath(clip)
        painter.fillRect(rect, QColor(PALETTE["card_alt"]))
        if not self._wdl or sum(self._wdl) == 0:
            return
        total = sum(self._wdl)
        x = rect.x()
        for value, color in zip(
            self._wdl, (PALETTE["white_piece"], "#8b8781", "#0d0c0b"), strict=True
        ):
            width = rect.width() * value / total
            painter.fillRect(QRectF(x, rect.y(), width, rect.height()), QColor(color))
            x += width


class Segmented(QWidget):
    """Choix exclusif en boutons accolés.

    ``changed`` n'est émis que sur action de l'utilisateur, pas par ``set_value``.
    """

    changed = Signal(object)

    def __init__(
        self, options: list[tuple[str, object]], *, object_names: list[str] | None = None
    ) -> None:
        super().__init__()
        self.setObjectName("Transparent")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._values: dict[int, object] = {}
        for index, (text, value) in enumerate(options):
            button = QPushButton(text)
            button.setObjectName(object_names[index] if object_names else "Segment")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            if index == 0:
                button.setProperty("position", "first")
            elif index == len(options) - 1:
                button.setProperty("position", "last")
            self._group.addButton(button, index)
            self._values[index] = value
            layout.addWidget(button)
        self._group.button(0).setChecked(True)
        self._group.idClicked.connect(lambda index: self.changed.emit(self._values[index]))

    def value(self) -> object:
        return self._values[self._group.checkedId()]

    def set_value(self, value: object) -> None:
        for index, candidate in self._values.items():
            if candidate == value:
                self._group.button(index).setChecked(True)

    def button(self, value: object) -> QPushButton:
        for index, candidate in self._values.items():
            if candidate == value:
                return self._group.button(index)
        raise KeyError(value)


class ElidedLabel(QLabel):
    """Libellé d'une ligne, terminé par « … » s'il est trop long pour la place disponible."""

    def __init__(self) -> None:
        super().__init__()
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self._elide()

    def resizeEvent(self, event) -> None:  # noqa: N802 (API Qt)
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        self.setText(metrics.elidedText(self._full, Qt.TextElideMode.ElideRight, self.width()))


class LineRow(QFrame):
    """Une ligne d'analyse cliquable : rang, coup, évaluation, suite."""

    clicked = Signal(int)

    def __init__(self, index: int) -> None:
        super().__init__()
        self.setObjectName("LineRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._index = index
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        self.badge = QLabel()
        self.badge.setFixedSize(12, 12)
        self.san_label = QLabel()
        self.san_label.setStyleSheet("font-weight: 700; font-size: 11pt;")
        self.eval_label = QLabel()
        self.eval_label.setStyleSheet("font-weight: 700;")
        self.pv_label = ElidedLabel()
        self.pv_label.setObjectName("Muted")
        layout.addWidget(self.badge)
        layout.addWidget(self.san_label)
        layout.addWidget(self.pv_label, stretch=1)
        layout.addWidget(self.eval_label)

    def set_line(
        self, move: str, continuation: str, evaluation: str, color: str, selected: bool
    ) -> None:
        self.badge.setStyleSheet(f"background: {color}; border-radius: 6px;")
        self.san_label.setText(move)
        self.eval_label.setText(evaluation)
        self.pv_label.set_full_text(continuation)
        self.setToolTip(f"{move}  {continuation}")
        self.setProperty("selected", "true" if selected else "false")
        restyle(self)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)


class Banner(QFrame):
    """Bandeau d'alerte avec boutons d'action."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Banner")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 10, 10)
        layout.setSpacing(12)
        self._icon = QLabel()
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._actions = QHBoxLayout()
        self._actions.setSpacing(8)
        layout.addWidget(self._icon, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._text, stretch=1)
        layout.addLayout(self._actions)
        self.hide()

    def show_message(self, kind: str, text: str, actions: Iterable[QPushButton] = ()) -> None:
        colors = {
            "warning": PALETTE["warning"],
            "danger": PALETTE["danger"],
            "info": PALETTE["info"],
        }
        name = {"warning": "alert", "danger": "alert", "info": "info"}[kind]
        self._icon.setPixmap(icon(name, colors[kind], 20).pixmap(20, 20))
        self._text.setText(text)
        while self._actions.count():
            item = self._actions.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for button in actions:
            self._actions.addWidget(button)
        self.setProperty("kind", kind)
        restyle(self)
        self.show()


class DropHint(QFrame):
    """État vide : invite à coller ou glisser une capture."""

    paste_requested = Signal()
    open_requested = Signal()
    play_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("DropHint")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)
        layout.addStretch(1)
        art = QLabel()
        art.setPixmap(icon("image", PALETTE["muted"], 56).pixmap(56, 56))
        art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Collez une capture de l'échiquier")
        title.setObjectName("DropTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle = QLabel()
        self.subtitle.setObjectName("Muted")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setWordWrap(True)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        paste = QPushButton(icon("paste", PALETTE["accent_text"]), "  Coller  (Ctrl+V)")
        paste.setObjectName("Primary")
        paste.clicked.connect(self.paste_requested)
        open_ = QPushButton(icon("open"), "  Ouvrir…")
        open_.clicked.connect(self.open_requested)
        play = QPushButton(icon("play"), "  Jouer contre l'IA")
        play.clicked.connect(self.play_requested)
        for button in (paste, open_, play):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addWidget(art)
        layout.addWidget(title)
        layout.addWidget(self.subtitle)
        layout.addSpacing(8)
        layout.addLayout(buttons)
        layout.addStretch(1)
        self.set_calibrated(True)

    def set_calibrated(self, calibrated: bool) -> None:
        if calibrated:
            self.subtitle.setText(
                "Faites une capture avec l'outil d'Ubuntu (Impr. écran), puis collez-la ici, "
                "glissez-déposez l'image ou ouvrez un fichier.\nLe trait, l'orientation et le "
                "meilleur coup sont trouvés automatiquement."
            )
        else:
            self.subtitle.setText(
                "Première utilisation : commencez par une capture de la POSITION INITIALE, dans "
                "votre thème Chess.com habituel.\nChessio y apprend vos pièces et vos couleurs, "
                "une fois pour toutes."
            )


class BoardPanel(QWidget):
    """Échiquier carré, barre d'évaluation alignée à sa gauche, invite superposée."""

    GAP = 10

    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.eval_bar = EvalBar()
        self.board = BoardView()
        self.hint = DropHint()
        for child in (self.eval_bar, self.board, self.hint):
            child.setParent(self)

    def resizeEvent(self, event) -> None:  # noqa: N802
        bar = self.eval_bar.width()
        side = max(100, min(self.height(), self.width() - bar - self.GAP))
        x = (self.width() - side - bar - self.GAP) // 2
        y = (self.height() - side) // 2
        self.eval_bar.setGeometry(x, y, bar, side)
        self.board.setGeometry(x + bar + self.GAP, y, side, side)
        self.hint.setGeometry(x, y, side + bar + self.GAP, side)

    def show_hint(self, visible: bool) -> None:
        self.hint.setVisible(visible)
        self.eval_bar.setVisible(not visible)
        self.board.setVisible(not visible)


class Toast(QLabel):
    """Message bref en bas de la fenêtre, qui s'efface tout seul."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("Toast")
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, seconds: float = 3.5) -> None:
        self.setText(text)
        self.adjustSize()
        parent = self.parentWidget()
        self.move((parent.width() - self.width()) // 2, parent.height() - self.height() - 24)
        self.raise_()
        self.show()
        self._timer.start(int(seconds * 1000))
