"""Match entre deux moteurs UCI : mesure de la force du moteur maison (section 9).

Chaque ouverture est jouée deux fois, couleurs inversées. Stockfish peut être
bridé à un Elo donné (``UCI_LimitStrength``) : le score obtenu contre lui donne
une estimation de l'Elo du moteur testé, avec un intervalle de confiance.

Le déroulé est communiqué coup par coup (``on_move``) et partie par partie
(``on_game``) : la ligne de commande l'affiche en texte, la fenêtre « match en
direct » sur un échiquier.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import chess
import chess.engine
import chess.pgn

from chessio.engine import HOMEMADE, find_stockfish, homemade_command

# Ouvertures courantes et équilibrées, pour varier les parties.
OPENINGS = (
    ("Partie espagnole", "e4 e5 Nf3 Nc6 Bb5"),
    ("Sicilienne ouverte", "e4 c5 Nf3 d6 d4 cxd4 Nxd4"),
    ("Gambit dame refusé", "d4 d5 c4 e6 Nc3 Nf6"),
    ("Française", "e4 e6 d4 d5 Nc3"),
    ("Caro-Kann", "e4 c6 d4 d5 Nc3 dxe4 Nxe4"),
    ("Est-indienne", "d4 Nf6 c4 g6 Nc3 Bg7"),
    ("Anglaise", "c4 e5 Nc3 Nf6"),
    ("Catalane inversée", "Nf3 d5 g3 Nf6 Bg2"),
    ("Italienne", "e4 e5 Nf3 Nc6 Bc4 Bc5"),
    ("Système de Londres", "d4 d5 Bf4 Nf6 e3"),
)
MAX_PLIES = 300  # au-delà : nulle par adjudication

TERMINATIONS = {
    chess.Termination.CHECKMATE: "échec et mat",
    chess.Termination.STALEMATE: "pat",
    chess.Termination.INSUFFICIENT_MATERIAL: "matériel insuffisant",
    chess.Termination.SEVENTYFIVE_MOVES: "règle des 75 coups",
    chess.Termination.FIVEFOLD_REPETITION: "répétition",
    chess.Termination.FIFTY_MOVES: "règle des 50 coups",
    chess.Termination.THREEFOLD_REPETITION: "répétition",
}


def engine_command(name: str) -> list[str]:
    """« maison », « stockfish » ou chemin d'un moteur UCI."""
    if name == HOMEMADE:
        return homemade_command()
    if name == "stockfish":
        return [str(find_stockfish())]
    return [name]


def elo_from_score(score: float) -> float:
    score = min(max(score, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / score - 1)


@dataclass(frozen=True)
class MatchSettings:
    tested: str = HOMEMADE
    reference: str = "stockfish"
    elo: int | None = None  # Elo imposé à l'adversaire
    games: int = 20
    seconds: float = 0.3  # par coup


@dataclass(frozen=True)
class MoveEvent:
    game: int  # numéro de la partie, à partir de 1
    board: chess.Board  # position après le coup
    move: chess.Move
    san: str
    score: chess.engine.PovScore | None  # évaluation annoncée par le moteur qui a joué
    tested_white: bool


@dataclass(frozen=True)
class GameResult:
    number: int
    opening: str
    result: str  # « 1-0 », « 0-1 » ou « 1/2-1/2 »
    tested_white: bool
    tested_points: float
    plies: int
    termination: str
    game: chess.pgn.Game


@dataclass
class Summary:
    tested: str
    reference: str
    elo: int | None
    points: list[float] = field(default_factory=list)

    @property
    def played(self) -> int:
        return len(self.points)

    @property
    def score(self) -> float:
        return sum(self.points) / self.played if self.points else 0.5

    def counts(self) -> tuple[int, int, int]:
        wins, draws = self.points.count(1.0), self.points.count(0.5)
        return wins, draws, self.played - wins - draws

    def elo_interval(self) -> tuple[float, float, float]:
        """(écart estimé, borne basse, borne haute) à 95 %, contre l'adversaire."""
        n, mean = self.played, self.score
        deviation = math.sqrt(sum((p - mean) ** 2 for p in self.points) / n) / math.sqrt(n)
        return (
            elo_from_score(mean),
            elo_from_score(mean - 1.96 * deviation),
            elo_from_score(mean + 1.96 * deviation),
        )

    def text(self) -> str:
        wins, draws, losses = self.counts()
        lines = [
            f"Score {sum(self.points):.1f}/{self.played} ({100 * self.score:.0f} %) : "
            f"{wins} gains, {draws} nulles, {losses} défaites"
        ]
        if self.played:
            diff, low, high = self.elo_interval()
            lines.append(
                f"Écart d'Elo estimé : {diff:+.0f} (intervalle à 95 % : {low:+.0f} à {high:+.0f})"
            )
            if self.elo is not None:
                lines.append(
                    f"Elo estimé du moteur testé : {self.elo + diff:.0f} "
                    f"({self.elo + low:.0f} à {self.elo + high:.0f})"
                )
        return "\n".join(lines)


class Match:
    """Joue un match ; ``stop()`` l'interrompt à la fin du coup en cours."""

    def __init__(self, settings: MatchSettings) -> None:
        self.settings = settings
        self.stop_requested = False
        self.names = (settings.tested, settings.reference)

    def stop(self) -> None:
        self.stop_requested = True

    def run(
        self,
        on_move: Callable[[MoveEvent], None] | None = None,
        on_game: Callable[[GameResult], None] | None = None,
    ) -> Summary:
        settings = self.settings
        summary = Summary(settings.tested, settings.reference, settings.elo)
        tested = chess.engine.SimpleEngine.popen_uci(engine_command(settings.tested))
        reference = chess.engine.SimpleEngine.popen_uci(engine_command(settings.reference))
        try:
            if settings.elo is not None:
                reference.configure({"UCI_LimitStrength": True, "UCI_Elo": settings.elo})
            if "Threads" in reference.options:
                reference.configure({"Threads": 1})
            label = reference.id.get("name", settings.reference)
            if settings.elo is not None:
                label += f" ({settings.elo} Elo)"
            self.names = (tested.id.get("name", settings.tested), label)
            summary.tested, summary.reference = self.names
            for index in range(settings.games):
                if self.stop_requested:
                    break
                result = self._play(index, tested, reference, on_move)
                if result is None:
                    break
                summary.points.append(result.tested_points)
                if on_game is not None:
                    on_game(result)
        finally:
            tested.quit()
            reference.quit()
        return summary

    def _play(
        self,
        index: int,
        tested: chess.engine.SimpleEngine,
        reference: chess.engine.SimpleEngine,
        on_move: Callable[[MoveEvent], None] | None,
    ) -> GameResult | None:
        opening_name, opening = OPENINGS[(index // 2) % len(OPENINGS)]
        tested_white = index % 2 == 0
        white, black = (tested, reference) if tested_white else (reference, tested)
        board = chess.Board()
        for san in opening.split():
            board.push_san(san)
        for engine in (white, black):
            engine.protocol.send_line("ucinewgame")  # parties indépendantes
        limit = chess.engine.Limit(time=self.settings.seconds)
        while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
            if self.stop_requested:
                return None
            engine = white if board.turn == chess.WHITE else black
            played = engine.play(board, limit, info=chess.engine.INFO_SCORE)
            san = board.san(played.move)
            board.push(played.move)
            if on_move is not None:
                on_move(
                    MoveEvent(
                        game=index + 1,
                        board=board.copy(),
                        move=played.move,
                        san=san,
                        score=played.info.get("score"),
                        tested_white=tested_white,
                    )
                )
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "1/2-1/2"
        termination = TERMINATIONS.get(outcome.termination, "fin") if outcome else "limite de coups"
        points = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "Match Chessio"
        game.headers["Round"] = str(index + 1)
        game.headers["White"], game.headers["Black"] = (
            self.names if tested_white else self.names[::-1]
        )
        game.headers["Result"] = result
        return GameResult(
            number=index + 1,
            opening=opening_name,
            result=result,
            tested_white=tested_white,
            tested_points=points if tested_white else 1 - points,
            plies=board.ply(),
            termination=termination,
            game=game,
        )
