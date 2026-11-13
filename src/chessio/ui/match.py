"""Fenêtre « match en direct » : deux moteurs s'affrontent sous vos yeux."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import chess
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from chessio.engine import format_score
from chessio.match import GameResult, Match, MatchSettings, MoveEvent, Summary
from chessio.ui.notation import figurine_line
from chessio.ui.style import PALETTE, app_icon, icon
from chessio.ui.widgets import BoardPanel, caption, card

RESULT_COLORS = {1.0: PALETTE["accent"], 0.5: PALETTE["muted"], 0.0: PALETTE["danger"]}
RESULT_WORDS = {1.0: "gain", 0.5: "nulle", 0.0: "défaite"}


def points_text(value: float) -> str:
    whole, half = int(value), value - int(value) >= 0.5
    return f"{whole}½" if half and whole else ("½" if half else str(whole))


class MatchWorker(QObject):
    """Joue le match hors du fil de l'interface, au rythme choisi."""

    moved = Signal(object)  # MoveEvent
    game_over = Signal(object)  # GameResult
    finished = Signal(object)  # Summary
    failed = Signal(str)

    def __init__(self, settings: MatchSettings, delay: float, pgn: Path | None) -> None:
        super().__init__()
        self.match = Match(settings)
        self.delay = delay  # secondes minimum entre deux coups affichés
        self.pgn = pgn
        self._running = threading.Event()
        self._running.set()

    @Slot()
    def run(self) -> None:
        try:
            summary = self.match.run(on_move=self._on_move, on_game=self._on_game)
            self.finished.emit(summary)
        except Exception as exc:  # noqa: BLE001 — remonté à l'utilisateur
            self.failed.emit(str(exc))

    def _on_move(self, event: MoveEvent) -> None:
        self.moved.emit(event)
        end = time.monotonic() + self.delay
        while not self.match.stop_requested and (
            time.monotonic() < end or not self._running.is_set()
        ):
            time.sleep(0.02)

    def _on_game(self, result: GameResult) -> None:
        if self.pgn is not None:
            with self.pgn.open("a", encoding="utf-8") as fh:
                print(result.game, file=fh, end="\n\n")
        self.game_over.emit(result)

    def pause(self, paused: bool) -> None:
        if paused:
            self._running.clear()
        else:
            self._running.set()

    def stop(self) -> None:
        self.match.stop()
        self._running.set()


class PlayerBar(QFrame):
    """Nom d'un joueur au-dessus ou au-dessous de l'échiquier, et s'il réfléchit."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Card")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.color = QLabel()
        self.name = QLabel()
        self.name.setStyleSheet("font-weight: 700; font-size: 11pt;")
        self.state = QLabel()
        self.state.setObjectName("Hint")
        layout.addWidget(self.color)
        layout.addWidget(self.name)
        layout.addStretch(1)
        layout.addWidget(self.state)

    def set_player(self, name: str, white: bool, thinking: bool) -> None:
        self.color.setText("○" if white else "●")
        self.name.setText(name)
        self.state.setText("réfléchit…" if thinking else "")


class MatchWindow(QMainWindow):
    def __init__(self, settings: MatchSettings, delay: float = 0.5, pgn: Path | None = None):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("Chessio — match en direct")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1000, 700)
        self.summary = Summary(settings.tested, settings.reference, settings.elo)
        self.current_game = 0
        self._build(delay)
        self._start(delay, pgn)

    # --- construction -------------------------------------------------------------

    def _build(self, delay: float) -> None:
        root = QWidget()
        body = QHBoxLayout(root)
        body.setContentsMargins(18, 16, 18, 16)
        body.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.top_bar = PlayerBar()
        self.panel = BoardPanel()
        self.panel.show_hint(False)
        self.bottom_bar = PlayerBar()
        left.addWidget(self.top_bar)
        left.addWidget(self.panel, stretch=1)
        left.addWidget(self.bottom_bar)
        body.addLayout(left, stretch=1)

        column = QVBoxLayout()
        column.setSpacing(12)
        column.addWidget(self._score_card())
        column.addWidget(self._game_card(), stretch=1)
        column.addWidget(self._games_card(), stretch=1)
        column.addWidget(self._controls_card(delay))
        side = QWidget()
        side.setLayout(column)
        side.setFixedWidth(400)
        body.addWidget(side)
        self.setCentralWidget(root)

    def _score_card(self) -> QFrame:
        frame, layout = card(spacing=6)
        layout.addWidget(caption("Score"))
        self.score_label = QLabel("0 – 0")
        self.score_label.setStyleSheet("font-size: 26pt; font-weight: 800;")
        self.names_label = QLabel("")
        self.names_label.setObjectName("Muted")
        self.names_label.setWordWrap(True)
        self.progress_label = QLabel("Démarrage des moteurs…")
        self.progress_label.setObjectName("Hint")
        self.elo_label = QLabel("")
        self.elo_label.setStyleSheet(f"color: {PALETTE['accent']}; font-weight: 700;")
        self.elo_label.setWordWrap(True)
        for widget in (self.score_label, self.names_label, self.progress_label, self.elo_label):
            layout.addWidget(widget)
        return frame

    def _game_card(self) -> QFrame:
        frame, layout = card(spacing=6)
        layout.addWidget(caption("Partie en cours"))
        self.game_label = QLabel("")
        self.game_label.setStyleSheet("font-weight: 700;")
        self.moves = QTextBrowser()
        self.moves.setStyleSheet(
            f"background: {PALETTE['surface']}; border: none; border-radius: 8px; padding: 6px;"
        )
        layout.addWidget(self.game_label)
        layout.addWidget(self.moves, stretch=1)
        return frame

    def _games_card(self) -> QFrame:
        frame, layout = card(spacing=6)
        layout.addWidget(caption("Parties jouées"))
        self.games_list = QVBoxLayout()
        self.games_list.setSpacing(4)
        self.games_list.addStretch(1)
        holder = QWidget()
        holder.setObjectName("Transparent")
        holder.setLayout(self.games_list)
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(scroll, stretch=1)
        return frame

    def _controls_card(self, delay: float) -> QFrame:
        frame, layout = card(spacing=8)
        row = QHBoxLayout()
        row.addWidget(QLabel("Délai entre les coups"))
        row.addStretch(1)
        self.delay_value = QLabel(f"{delay:.1f} s".replace(".", ","))
        self.delay_value.setStyleSheet("font-weight: 700;")
        row.addWidget(self.delay_value)
        self.delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.delay_slider.setRange(0, 30)  # dixièmes de seconde
        self.delay_slider.setValue(int(delay * 10))
        self.delay_slider.valueChanged.connect(self._on_delay)
        buttons = QHBoxLayout()
        self.pause_button = QPushButton(icon("stop"), "  Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._on_pause)
        self.stop_button = QPushButton(icon("close"), "  Arrêter le match")
        self.stop_button.clicked.connect(self._stop)
        for button in (self.pause_button, self.stop_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            buttons.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(self.delay_slider)
        layout.addLayout(buttons)
        return frame

    # --- déroulé ------------------------------------------------------------------

    def _start(self, delay: float, pgn: Path | None) -> None:
        self.worker_thread = QThread(self)
        self.worker = MatchWorker(self.settings, delay, pgn)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.moved.connect(self._on_move)
        self.worker.game_over.connect(self._on_game)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker_thread.start()

    def _names(self) -> tuple[str, str]:
        return self.worker.match.names

    def _on_move(self, event: MoveEvent) -> None:
        tested, reference = self._names()
        board = event.board
        if event.game != self.current_game:
            self.current_game = event.game
            self.summary.tested, self.summary.reference = tested, reference
            self._update_score()
        white_bottom = event.tested_white  # le moteur testé toujours en bas
        white_name, black_name = (tested, reference) if event.tested_white else (reference, tested)
        bottom_white = white_bottom
        self.bottom_bar.set_player(
            white_name if bottom_white else black_name,
            bottom_white,
            board.turn == (chess.WHITE if bottom_white else chess.BLACK),
        )
        self.top_bar.set_player(
            black_name if bottom_white else white_name,
            not bottom_white,
            board.turn == (chess.BLACK if bottom_white else chess.WHITE),
        )
        self.panel.board.set_position(board, white_bottom=white_bottom, lastmove=event.move)
        if event.score is not None:
            mate = event.score.white().mate()
            text = f"M{abs(mate)}" if mate is not None else format_score(event.score).lstrip("+")
            expectation = event.score.white().wdl(model="sf").expectation()
            self.panel.eval_bar.set_eval(expectation, text, white_bottom)
        opening = self._opening_name(event.game)
        self.game_label.setText(
            f"Partie {event.game} / {self.settings.games} · {opening} · "
            f"coup {board.fullmove_number}"
        )
        self.moves.setPlainText(figurine_line(chess.Board(), board.move_stack))
        self.moves.verticalScrollBar().setValue(self.moves.verticalScrollBar().maximum())

    def _opening_name(self, game: int) -> str:
        from chessio.match import OPENINGS

        return OPENINGS[((game - 1) // 2) % len(OPENINGS)][0]

    def _on_game(self, result: GameResult) -> None:
        self.summary.points.append(result.tested_points)
        tested, reference = self._names()
        color = RESULT_COLORS[result.tested_points]
        side = "Blancs" if result.tested_white else "Noirs"
        row = QLabel(
            f"<span style='color:{color}; font-weight:700'>●</span>&nbsp; "
            f"<b>{result.number}.</b> {result.result} — {RESULT_WORDS[result.tested_points]} "
            f"({side}), {result.termination}, {result.plies} demi-coups"
        )
        row.setToolTip(f"{result.opening} : {tested} contre {reference}")
        self.games_list.insertWidget(self.games_list.count() - 1, row)
        self._update_score()
        print(
            f"partie {result.number:2d} : {result.result:7s} ({side}, {result.termination}) "
            f"— total {sum(self.summary.points):.1f}/{self.summary.played}",
            flush=True,
        )

    def _update_score(self) -> None:
        tested, reference = self._names()
        played = self.summary.played
        mine = sum(self.summary.points)
        self.score_label.setText(f"{points_text(mine)} – {points_text(played - mine)}")
        self.names_label.setText(f"{tested}  contre  {reference}")
        self.progress_label.setText(
            f"{played} partie{'s' if played > 1 else ''} terminée{'s' if played > 1 else ''} "
            f"sur {self.settings.games} · {self.settings.seconds:.1f} s par coup".replace(".", ",")
        )
        if played and self.settings.elo is not None:
            diff, low, high = self.summary.elo_interval()
            elo = self.settings.elo
            self.elo_label.setText(
                f"Elo estimé de {tested} : {elo + diff:.0f}  (entre {elo + low:.0f} et "
                f"{elo + high:.0f}, à 95 %)"
            )

    def _on_finished(self, summary: Summary) -> None:
        self.summary = summary
        self._update_score()
        self.progress_label.setText("Match terminé.")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.top_bar.state.setText("")
        self.bottom_bar.state.setText("")
        print("\n" + summary.text(), flush=True)

    def _on_failed(self, message: str) -> None:
        self.progress_label.setText(f"Erreur : {message}")
        print(f"Erreur : {message}", flush=True)

    def _on_delay(self, value: int) -> None:
        self.worker.delay = value / 10
        self.delay_value.setText(f"{value / 10:.1f} s".replace(".", ","))

    def _on_pause(self, paused: bool) -> None:
        self.worker.pause(paused)
        self.pause_button.setText("  Reprendre" if paused else "  Pause")
        self.pause_button.setIcon(icon("play" if paused else "stop"))

    def _stop(self) -> None:
        self.worker.stop()
        self.progress_label.setText("Arrêt après le coup en cours…")

    def closeEvent(self, event) -> None:  # noqa: N802 (API Qt)
        self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait(10_000)
        super().closeEvent(event)
