"""Pilotage d'un moteur UCI (Stockfish ou moteur maison) dans un processus séparé (EF-08, EF-10).

Le moteur tourne toujours dans son propre processus : python-chess le lance
et lui parle en UCI. On change donc de moteur en changeant d'exécutable.

L'analyse est progressive : le meilleur coup provisoire est communiqué à chaque
approfondissement, et l'analyse s'interrompt dès que la position n'intéresse
plus (nouvelle capture, correction d'une case…).
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import chess
import chess.engine

from chessio.position import ensure_valid

ENGINE_ENV = "CHESSIO_STOCKFISH"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ENF-05 : temps de réflexion réglable de 0,1 à 10 s, 2 s par défaut.
MIN_TIME = 0.1
MAX_TIME = 10.0
DEFAULT_TIME = 2.0
DEFAULT_HASH_MB = 256
UPDATE_INTERVAL = 0.1  # secondes entre deux résultats provisoires


class EngineNotFoundError(FileNotFoundError):
    """Aucun exécutable de moteur n'a été trouvé."""


def find_stockfish() -> Path:
    """Cherche Stockfish : variable ``CHESSIO_STOCKFISH``, puis ``engines/``, puis le PATH."""
    candidates: list[Path] = []
    if env := os.environ.get(ENGINE_ENV):
        candidates.append(Path(env))
    candidates.append(PROJECT_ROOT / "engines" / "stockfish")
    if found := shutil.which("stockfish"):
        candidates.append(Path(found))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise EngineNotFoundError(
        "Stockfish introuvable. Lancez scripts/fetch_stockfish.sh ou définissez "
        f"{ENGINE_ENV}=/chemin/vers/stockfish."
    )


def default_threads() -> int:
    """Tous les cœurs moins un, pour laisser l'interface réactive (section 7)."""
    return max(1, (os.cpu_count() or 2) - 1)


def format_score(score: chess.engine.PovScore) -> str:
    """Évaluation du point de vue des Blancs : ``+0.35``, ``-1.20``, ``#3``, ``#-2``."""
    white = score.white()
    mate = white.mate()
    if mate is not None:
        return f"#{mate}"
    cp = white.score()
    assert cp is not None
    return f"{cp / 100:+.2f}"


@dataclass(frozen=True)
class Line:
    """Une ligne proposée par le moteur (un coup candidat et sa suite)."""

    move: chess.Move
    san: str
    score: chess.engine.PovScore
    pv: tuple[chess.Move, ...]
    pv_san: str
    depth: int
    wdl: chess.engine.Wdl | None = None  # chances (‰) du point de vue des Blancs

    @property
    def uci(self) -> str:
        return self.move.uci()

    @property
    def evaluation(self) -> str:
        return format_score(self.score)

    @property
    def white_wdl(self) -> chess.engine.Wdl:
        """Gain / nul / perte des Blancs (‰) : donnés par le moteur, sinon estimés."""
        return self.wdl if self.wdl is not None else self.score.white().wdl(model="sf")

    @property
    def white_expectation(self) -> float:
        """Espérance de points des Blancs, de 0 (perdu) à 1 (gagné)."""
        return self.white_wdl.expectation()


@dataclass(frozen=True)
class Analysis:
    """Résultat d'une analyse : lignes triées de la meilleure à la moins bonne."""

    fen: str
    lines: tuple[Line, ...]
    elapsed: float
    outcome: str | None = None  # « mat » ou « pat » si la partie est déjà finie
    partial: bool = False  # résultat provisoire, l'analyse continue
    stopped: bool = False  # analyse interrompue avant la fin du temps imparti

    @property
    def best(self) -> Line | None:
        return self.lines[0] if self.lines else None


class Engine:
    """Moteur UCI dans un processus séparé. À utiliser avec ``with``."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        threads: int | None = None,
        hash_mb: int = DEFAULT_HASH_MB,
    ) -> None:
        self.path = Path(path) if path is not None else find_stockfish()
        self._engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
        options = {
            "Threads": threads or default_threads(),
            "Hash": hash_mb,
            "UCI_ShowWDL": True,  # chances de gain / nul / perte
        }
        self._engine.configure(
            {name: value for name, value in options.items() if name in self._engine.options}
        )
        # Échauffement : la première recherche paie l'initialisation paresseuse du
        # moteur (table de hachage, réseau NNUE), environ 0,5 s. On la paie au
        # démarrage plutôt que sur la première demande de l'utilisateur (ENF-04).
        self._engine.analyse(chess.Board(), chess.engine.Limit(depth=1))

    @property
    def name(self) -> str:
        return self._engine.id.get("name", self.path.name)

    def analyse(
        self,
        board: chess.Board,
        *,
        time_limit: float = DEFAULT_TIME,
        multipv: int = 1,
        on_update: Callable[[Analysis], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Analysis:
        """Calcule les ``multipv`` meilleurs coups en ``time_limit`` secondes.

        ``on_update`` reçoit des résultats provisoires au fil de l'approfondissement ;
        ``should_stop`` permet d'interrompre l'analyse (vérifié à chaque information
        du moteur, soit plusieurs fois par seconde).

        Lève :class:`~chessio.position.InvalidPositionError` sur une position
        douteuse : on ne propose jamais de coup sur une position invalide.
        """
        if not MIN_TIME <= time_limit <= MAX_TIME:
            raise ValueError(f"Temps de réflexion hors limites ({MIN_TIME} à {MAX_TIME} s)")
        if multipv < 1:
            raise ValueError("multipv doit valoir au moins 1")
        ensure_valid(board)

        fen = board.fen()
        if board.is_checkmate() or board.is_stalemate():
            outcome = "mat" if board.is_checkmate() else "pat"
            return Analysis(fen=fen, lines=(), elapsed=0.0, outcome=outcome)

        start = time.perf_counter()
        lines: dict[int, Line] = {}
        stopped = False
        last_update = 0.0
        limit = chess.engine.Limit(time=time_limit)
        with self._engine.analysis(board, limit, multipv=multipv) as running:
            for info in running:
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                if "pv" not in info or "score" not in info:
                    continue
                lines[info.get("multipv", 1)] = _to_line(board, info)
                now = time.perf_counter()
                if on_update is not None and now - last_update >= UPDATE_INTERVAL:
                    last_update = now
                    on_update(Analysis(fen, _ordered(lines), now - start, partial=True))
            if not stopped:
                # Dernières lignes complètes, dans l'ordre du moteur.
                final = [info for info in running.multipv if "pv" in info and "score" in info]
                if final:
                    lines = {i: _to_line(board, info) for i, info in enumerate(final, start=1)}
        return Analysis(fen, _ordered(lines), time.perf_counter() - start, stopped=stopped)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> Engine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _ordered(lines: dict[int, Line]) -> tuple[Line, ...]:
    return tuple(lines[index] for index in sorted(lines))


def _to_line(board: chess.Board, info: chess.engine.InfoDict) -> Line:
    pv = tuple(info["pv"])
    wdl = info.get("wdl")
    return Line(
        move=pv[0],
        san=board.san(pv[0]),
        score=info["score"],
        pv=pv,
        pv_san=board.variation_san(pv),
        depth=info.get("depth", 0),
        wdl=wdl.white() if wdl is not None else None,
    )
