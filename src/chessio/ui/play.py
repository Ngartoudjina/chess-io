"""Jouer contre l'IA : Stockfish à un niveau choisi ou le moteur maison, avec un coach.

Le coach (Stockfish à pleine force) donne un indice sur demande et l'évaluation
de la position ; l'adversaire joue au niveau choisi (``UCI_LimitStrength``).
"""

from __future__ import annotations

import random

import chess
import chess.engine
from PySide6.QtCore import QObject, QPoint, Qt, QThread, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from chessio import coach
from chessio.engine import HOMEMADE, Engine
from chessio.match import engine_command
from chessio.ui.notation import figurine_line
from chessio.ui.style import ARROW_NAMES, PALETTE, app_icon, icon, piece_icon
from chessio.ui.widgets import BoardPanel, Segmented, caption, card

# Niveaux proposés : (libellé, moteur, Elo imposé ou None pour la pleine force).
LEVELS = (
    ("Débutant · 1320", "stockfish", 1320),
    ("Club · 1600", "stockfish", 1600),
    ("Fort · 2000", "stockfish", 2000),
    ("Expert · 2400", "stockfish", 2400),
    ("Maître · 2800", "stockfish", 2800),
    ("Maximum", "stockfish", None),
    ("Moteur maison", HOMEMADE, None),
)
SELECTED_FILL = "#f6f66980"
TARGET_FILL = "#2563eb55"
RESULTS = {"1-0": "Les Blancs gagnent", "0-1": "Les Noirs gagnent", "1/2-1/2": "Partie nulle"}
REASONS = {
    chess.Termination.CHECKMATE: "échec et mat",
    chess.Termination.STALEMATE: "pat",
    chess.Termination.INSUFFICIENT_MATERIAL: "matériel insuffisant",
    chess.Termination.THREEFOLD_REPETITION: "répétition",
    chess.Termination.FIVEFOLD_REPETITION: "répétition",
    chess.Termination.FIFTY_MOVES: "règle des 50 coups",
    chess.Termination.SEVENTYFIVE_MOVES: "règle des 75 coups",
}


class PlayWorker(QObject):
    """Moteurs hors du fil de l'interface : l'adversaire et le coach."""

    ready = Signal(str)
    ai_moved = Signal(int, str)  # génération, coup UCI
    evaluated = Signal(int, object)  # génération, Analysis du coach
    coached = Signal(int, object, object)  # génération, Analysis (3 coups), ligne de menace
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.opponent: chess.engine.SimpleEngine | None = None
        self.coach: Engine | None = None
        self.latest = 0

    @Slot(str, object)
    def configure(self, choice: str, elo: object) -> None:
        try:
            if self.opponent is not None:
                self.opponent.quit()
            self.opponent = chess.engine.SimpleEngine.popen_uci(engine_command(choice))
            if elo is not None and "UCI_LimitStrength" in self.opponent.options:
                self.opponent.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
            if self.coach is None:
                self.coach = Engine()
            self.ready.emit(self.opponent.id.get("name", choice))
        except Exception as exc:  # noqa: BLE001 — remonté à l'utilisateur
            self.failed.emit(f"Moteur indisponible : {exc}")

    @Slot(int, str, float)
    def play(self, generation: int, fen: str, seconds: float) -> None:
        if self.opponent is None or generation != self.latest:
            return
        try:
            result = self.opponent.play(chess.Board(fen), chess.engine.Limit(time=seconds))
            if result.move is not None:
                self.ai_moved.emit(generation, result.move.uci())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    @Slot(int, str, float)
    def evaluate(self, generation: int, fen: str, seconds: float) -> None:
        if self.coach is None or generation != self.latest:
            return
        try:
            self.evaluated.emit(
                generation,
                self.coach.analyse(
                    chess.Board(fen),
                    time_limit=seconds,
                    should_stop=lambda: self.latest != generation,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    @Slot(int, str)
    def coach_position(self, generation: int, fen: str) -> None:
        """Analyse pour l'assistant : 3 meilleurs coups, et menace adverse (coup nul)."""
        if self.coach_engine_missing() or generation != self.latest:
            return
        try:
            board = chess.Board(fen)
            stop = lambda: self.latest != generation  # noqa: E731
            analysis = self.coach.analyse(board, time_limit=1.0, multipv=3, should_stop=stop)
            threat = None
            if not board.is_check() and not analysis.outcome:
                null = board.copy(stack=False)
                null.turn = not null.turn  # l'adversaire rejoue (sans historique de coup nul)
                null.ep_square = None
                if null.is_valid():
                    threat = self.coach.analyse(null, time_limit=0.5, should_stop=stop).best
            if generation == self.latest:
                self.coached.emit(generation, analysis, threat)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def coach_engine_missing(self) -> bool:
        return self.coach is None

    @Slot()
    def stop(self) -> None:
        if self.opponent is not None:
            self.opponent.quit()
            self.opponent = None
        if self.coach is not None:
            self.coach.close()
            self.coach = None


class PlayWindow(QMainWindow):
    request_configure = Signal(str, object)
    request_play = Signal(int, str, float)
    request_evaluate = Signal(int, str, float)
    request_coach = Signal(int, str)
    request_stop = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Chessio — jouer contre l'IA")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(980, 680)
        self.board = chess.Board()
        self.human = chess.WHITE
        self.selected: chess.Square | None = None
        self.hint: chess.Move | None = None
        self.generation = 0
        self.engines_ready = False
        self.best_now = None  # meilleure ligne du coach à votre tour
        self.review = None  # (position avant, coup joué, meilleure ligne) à juger
        self.feed: list = []  # conseils affichés
        self.game_over_shown = False
        self._build()
        self._start_worker()

    # --- construction ----------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        body = QHBoxLayout(root)
        body.setContentsMargins(18, 16, 18, 16)
        body.setSpacing(18)
        left = QVBoxLayout()
        self.panel = BoardPanel()
        self.panel.show_hint(False)
        self.panel.board.interactive = True
        self.panel.board.square_clicked.connect(self._on_square)
        self.status = QLabel("Démarrage des moteurs…")
        self.status.setObjectName("Status")
        self.moves = QLabel("")
        self.moves.setObjectName("Muted")
        self.moves.setToolTip("Derniers coups de la partie")
        left.addWidget(self.panel, stretch=1)
        left.addWidget(self.status)
        left.addWidget(self.moves)
        body.addLayout(left, stretch=1)

        # Colonne de droite dans une zone à défilement : sur un petit écran, rien n'est écrasé.
        side = QWidget()
        column = QVBoxLayout(side)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(12)
        # Carte « Partie » compacte : les réglages ne s'ouvrent qu'à la demande, pour
        # laisser la place à l'assistant.
        frame, layout = card(spacing=8)
        layout.addWidget(caption("Partie"))
        self.game_label = QLabel("")
        self.game_label.setWordWrap(True)
        self.game_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.game_label)
        buttons = QHBoxLayout()
        new_game = QPushButton(icon("play", PALETTE["accent_text"]), "  Nouvelle partie")
        new_game.setObjectName("Primary")
        new_game.clicked.connect(self.new_game)
        self.setup_button = QPushButton(icon("menu"), "  Couleur et niveau")
        self.setup_button.setCheckable(True)
        self.setup_button.toggled.connect(lambda shown: self.setup.setVisible(shown))
        buttons.addWidget(new_game)
        buttons.addWidget(self.setup_button)
        layout.addLayout(buttons)
        self.setup = QWidget()
        self.setup.setObjectName("Transparent")
        setup = QVBoxLayout(self.setup)
        setup.setContentsMargins(0, 4, 0, 0)
        setup.setSpacing(8)
        setup.addWidget(QLabel("Vos pièces"))
        self.color_choice = Segmented([("Blancs", "w"), ("Noirs", "b"), ("Au hasard", "?")])
        setup.addWidget(self.color_choice)
        setup.addWidget(QLabel("Niveau de l'IA"))
        self.level_choice = Segmented(
            [(label.split(" · ")[0], i) for i, (label, _, _) in enumerate(LEVELS[:4])]
        )
        self.level_choice2 = Segmented(
            [(label.split(" · ")[0], i) for i, (label, _, _) in enumerate(LEVELS) if i >= 4]
        )
        self.level_choice.set_value(1)
        self._uncheck(self.level_choice2)  # un seul niveau sélectionné
        self.level_choice.changed.connect(lambda _: self._pick(self.level_choice2))
        self.level_choice2.changed.connect(lambda _: self._pick(self.level_choice))
        self.level_label = QLabel()
        self.level_label.setObjectName("Hint")
        self.level_label.setWordWrap(True)
        setup.addWidget(self.level_choice)
        setup.addWidget(self.level_choice2)
        setup.addWidget(self.level_label)
        self.setup.setVisible(False)
        layout.addWidget(self.setup)
        column.addWidget(frame)

        frame, layout = card(spacing=8)
        layout.addWidget(caption("Assistant"))
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.eval_label = QLabel("")
        self.eval_label.setObjectName("Hint")
        self.eval_label.setWordWrap(True)
        layout.addWidget(self.eval_label)
        self.assistant = QTextBrowser()
        self.assistant.setMinimumHeight(200)
        self.assistant.setPlaceholderText(
            "Jouez un coup : l'assistant vous dira à chaque tour quoi faire, "
            "ce que prépare l'IA et ce qu'il faut éviter."
        )
        layout.addWidget(self.assistant, stretch=1)
        row = QHBoxLayout()
        self.hint_button = QPushButton(icon("info"), "  Indice (H)")
        self.hint_button.setToolTip("Montre le meilleur coup sur l'échiquier")
        self.hint_button.clicked.connect(self._ask_hint)
        undo = QPushButton(icon("flip"), "  Annuler (Ctrl+Z)")
        undo.setToolTip("Reprend votre dernier coup")
        undo.clicked.connect(self._undo)
        row.addWidget(self.hint_button)
        row.addWidget(undo)
        layout.addLayout(row)
        column.addWidget(frame, stretch=1)

        scroll = QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(410)
        body.addWidget(scroll)
        self.setCentralWidget(root)
        QShortcut(QKeySequence("H"), self, activated=self._ask_hint)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._undo)
        self._update_level_label()

    def _pick(self, other: Segmented) -> None:
        self._uncheck(other)
        self._update_level_label()

    @staticmethod
    def _uncheck(segmented: Segmented) -> None:
        group = segmented._group
        group.setExclusive(False)
        for button in group.buttons():
            button.setChecked(False)
        group.setExclusive(True)

    def _level(self) -> tuple[str, str, int | None]:
        for segmented in (self.level_choice, self.level_choice2):
            if segmented._group.checkedId() >= 0:
                return LEVELS[int(segmented.value())]
        return LEVELS[1]

    def _update_level_label(self) -> None:
        label, _, _ = self._level()
        self.level_label.setText(
            f"Niveau choisi : {label}. Cliquez sur « Nouvelle partie » pour l'appliquer."
        )

    def _start_worker(self) -> None:
        self.worker_thread = QThread(self)
        self.worker = PlayWorker()
        self.worker.moveToThread(self.worker_thread)
        self.request_configure.connect(self.worker.configure)
        self.request_play.connect(self.worker.play)
        self.request_evaluate.connect(self.worker.evaluate)
        self.request_coach.connect(self.worker.coach_position)
        self.worker.coached.connect(self._on_coached)
        self.request_stop.connect(self.worker.stop, Qt.ConnectionType.BlockingQueuedConnection)
        self.worker.ready.connect(self._on_ready)
        self.worker.ai_moved.connect(self._on_ai_move)
        self.worker.evaluated.connect(self._on_evaluated)
        self.worker.failed.connect(lambda message: self.status.setText(message))
        self.worker_thread.start()
        self.new_game()

    # --- déroulé -----------------------------------------------------------------

    def _next(self) -> int:
        self.generation += 1
        self.worker.latest = self.generation
        return self.generation

    def new_game(self) -> None:
        choice = self.color_choice.value()
        self.human = {"w": chess.WHITE, "b": chess.BLACK}.get(choice, random.choice(chess.COLORS))
        self.board = chess.Board()
        self.selected = self.hint = None
        self.game_over_shown = False
        self.engines_ready = False
        self._next()
        self._update_level_label()
        label, engine, elo = self._level()
        self.status.setText(f"Préparation de l'IA ({label})…")
        side = "Blancs" if self.human == chess.WHITE else "Noirs"
        self.game_label.setText(f"Vous avez les {side} · IA : {label}")
        self.request_configure.emit(engine, elo)
        self.setup_button.setChecked(False)
        self.summary_label.setText("")
        self.eval_label.setText("")
        self.assistant.setHtml("")
        self.feed = []
        self.review = None
        self._redraw()

    def _on_ready(self, name: str) -> None:
        self.engines_ready = True
        side = "Blancs" if self.human == chess.WHITE else "Noirs"
        self.status.setText(f"Vous jouez les {side} contre {name} ({self._level()[0]}).")
        self.game_label.setText(f"Vous avez les {side} · IA : {self._level()[0]}")
        self._after_move()

    def _after_move(self) -> None:
        self.hint = None
        self._redraw()
        if self._check_game_over():
            return
        generation = self._next()
        if self.board.turn != self.human and self.engines_ready:
            self.request_evaluate.emit(generation, self.board.fen(), 0.4)
            self.status.setText("L'IA réfléchit…")
            self.request_play.emit(generation, self.board.fen(), 0.6)
        elif self.engines_ready:
            self.best_now = None
            self.request_coach.emit(generation, self.board.fen())
            self.status.setText("À vous de jouer. L'assistant analyse…")

    def _on_ai_move(self, generation: int, uci: str) -> None:
        if generation != self.generation:
            return
        move = chess.Move.from_uci(uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            self._after_move()

    def _on_evaluated(self, generation: int, analysis) -> None:
        if generation != self.generation or analysis.best is None:
            return
        self._show_eval(analysis)
        if self.review is not None:  # votre dernier coup, jugé sur la position qui en résulte
            before, move, best_before = self.review
            self.review = None
            if best_before is not None:
                loss = coach._cp(best_before, self.human) - coach._cp(analysis.best, self.human)
                self.feed = [
                    coach.move_quality(max(0, loss), best_before, before, move, analysis.best)
                ]
                self._render_feed()

    def _on_coached(self, generation: int, analysis, threat_line) -> None:
        if generation != self.generation:
            return
        if analysis.best is None:
            return
        self._show_eval(analysis)
        self.best_now = analysis.best
        advices = list(self.feed)  # jugement de votre coup précédent
        if self.board.turn == self.human:
            advices += coach.advise(self.board, analysis.lines, threat_line)
        advices.sort(key=lambda a: coach.ORDER.get(a.kind, 9))  # la priorité d'abord
        self.feed = []
        self._render_feed(advices)
        self.status.setText("À vous de jouer.")

    def _render_feed(self, advices=None) -> None:
        colors = {
            "bien": PALETTE["accent"],
            "info": PALETTE["muted"],
            "attention": PALETTE["warning"],
            "danger": PALETTE["danger"],
        }
        blocks = []
        for advice in advices if advices is not None else self.feed:
            color = colors[advice.level]
            if advice.kind == "action" and advice.level == "info":
                color = PALETTE["info"]  # la priorité reste visible même sans urgence
            title = coach.TITLES.get(advice.kind, advice.kind)
            # « À faire maintenant » ressort : texte plus grand, fond teinté.
            key = advice.kind == "action"
            background = f" bgcolor='{PALETTE['card_alt']}'" if key else ""
            size = "11pt" if key else "10pt"
            blocks.append(
                "<table width='100%' cellspacing='0' cellpadding='0' style='margin-bottom:10px'>"
                f"<tr><td width='4' bgcolor='{color}'></td>"
                f"<td{background} style='padding:6px 10px'>"
                f"<span style='color:{color}; font-weight:700; font-size:9pt'>"
                f"{title.upper()}</span><br>"
                f"<span style='font-size:{size}'>{advice.text}</span></td></tr></table>"
            )
        self.assistant.setHtml("".join(blocks))

    def _show_eval(self, analysis) -> None:
        best = analysis.best
        mate = best.score.white().mate()
        text = f"M{abs(mate)}" if mate is not None else best.evaluation.lstrip("+")
        self.panel.eval_bar.set_eval(best.white_expectation, text, self.human == chess.WHITE)
        words, level = coach.summary(self.board, best, self.human)
        color = {
            "bien": PALETTE["accent"],
            "info": PALETTE["text"],
            "attention": PALETTE["warning"],
            "danger": PALETTE["danger"],
        }[level]
        self.summary_label.setText(words)
        self.summary_label.setStyleSheet(f"color: {color}; font-weight: 600; font-size: 11pt;")
        mine = best.score.pov(self.human)
        if mine.is_mate():
            self.eval_label.setText("")
        else:
            pawns = mine.score() / 100
            unit = "pions" if abs(pawns) >= 2 else "pion"
            self.eval_label.setText(
                f"Évaluation {pawns:+.1f} {unit} (positif : vous êtes mieux)".replace("-", "−")
            )

    def _ask_hint(self) -> None:
        if self.board.turn != self.human or self.board.is_game_over() or self.best_now is None:
            return
        self.hint = self.best_now.move
        self.status.setText(
            f"Indice : {self.board.san(self.hint)} — {coach.describe_move(self.board, self.hint)}."
        )
        self._redraw()

    def _undo(self) -> None:
        if len(self.board.move_stack) < 1:
            return
        self._next()  # annule une réponse de l'IA en attente
        self.board.pop()
        if self.board.turn != self.human and self.board.move_stack:
            self.board.pop()
        self.game_over_shown = False
        self.selected = None
        self._after_move()

    # --- coups de l'utilisateur ------------------------------------------------------

    def _on_square(self, square: chess.Square, position: QPoint) -> None:
        if self.board.turn != self.human or self.board.is_game_over() or not self.engines_ready:
            return
        piece = self.board.piece_at(square)
        if self.selected is not None:
            candidates = [
                m
                for m in self.board.legal_moves
                if m.from_square == self.selected and m.to_square == square
            ]
            if candidates:
                move = candidates[0]
                if len(candidates) > 1:  # promotion
                    move = self._choose_promotion(candidates, position)
                    if move is None:
                        return
                self.selected = None
                self.review = (self.board.copy(), move, self.best_now)
                self.board.push(move)
                self._after_move()
                return
        self.selected = square if piece is not None and piece.color == self.human else None
        self._redraw()

    def _choose_promotion(self, candidates: list[chess.Move], position: QPoint):
        menu = QMenu(self)
        for move in sorted(candidates, key=lambda m: -m.promotion):
            symbol = chess.piece_symbol(move.promotion)
            symbol = symbol.upper() if self.human == chess.WHITE else symbol
            menu.addAction(piece_icon(symbol), chess.piece_name(move.promotion)).setData(move)
        chosen = menu.exec(position)
        return chosen.data() if chosen is not None else None

    def _redraw(self) -> None:
        fill = {}
        if self.selected is not None:
            fill[self.selected] = SELECTED_FILL
            for move in self.board.legal_moves:
                if move.from_square == self.selected:
                    fill[move.to_square] = TARGET_FILL
        arrows = [(self.hint, ARROW_NAMES[0])] if self.hint is not None else []
        lastmove = self.board.peek() if self.board.move_stack else None
        self.panel.board.set_position(
            self.board,
            white_bottom=self.human == chess.WHITE,
            arrows=arrows,
            lastmove=lastmove,
            fill=fill,
        )
        tokens = figurine_line(chess.Board(), self.board.move_stack).split()
        text = " ".join(tokens[-15:])
        self.moves.setText(("… " if len(tokens) > 15 else "") + text)

    def _check_game_over(self) -> bool:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return False
        result = outcome.result()
        reason = REASONS.get(outcome.termination, "fin de partie")
        won = outcome.winner == self.human
        verdict = (
            "Bravo, vous gagnez !"
            if won
            else ("Nulle." if outcome.winner is None else "L'IA gagne.")
        )
        self.status.setText(f"{RESULTS[result]} ({reason}). {verdict}")
        if not self.game_over_shown:
            self.game_over_shown = True
            QMessageBox.information(
                self, "Fin de partie", f"{RESULTS[result]} : {reason}.\n{verdict}"
            )
        return True

    def closeEvent(self, event) -> None:  # noqa: N802 (API Qt)
        self._next()
        self.request_stop.emit()
        self.worker_thread.quit()
        self.worker_thread.wait(5000)
        super().closeEvent(event)
