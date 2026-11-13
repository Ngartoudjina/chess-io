"""Protocole UCI du moteur maison : ``python -m chessio.moteur``.

La recherche tourne dans un fil à part : le moteur répond à tout moment à
``isready`` et s'arrête sur ``stop``. ``go perft N`` affiche le perft détaillé
par coup, comme Stockfish.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

import chess

from chessio.moteur import NAME, VERSION
from chessio.moteur.perft import divide
from chessio.moteur.search import (
    DEFAULT_TT_ENTRIES,
    Limits,
    Searcher,
    SearchInfo,
    mate_in,
)

ENTRIES_PER_MB = 2_500  # ordre de grandeur pour un dictionnaire Python
DEFAULT_HASH_MB = DEFAULT_TT_ENTRIES // ENTRIES_PER_MB


def uci_score(score: int) -> str:
    mate = mate_in(score)
    return f"mate {mate}" if mate is not None else f"cp {score}"


def parse_limits(tokens: list[str]) -> Limits:
    """Arguments de « go » (temps en millisecondes côté UCI, en secondes ici)."""
    limits = Limits()
    seconds = {"wtime", "btime", "winc", "binc", "movetime"}
    integers = {"depth", "nodes", "movestogo"}
    index = 0
    while index < len(tokens):
        name = tokens[index]
        if name == "infinite":
            limits.infinite = True
        elif name in seconds | integers and index + 1 < len(tokens):
            value = int(tokens[index + 1])
            setattr(limits, name, value / 1000 if name in seconds else value)
            index += 1
        index += 1
    return limits


class UciEngine:
    def __init__(self, out: TextIO = sys.stdout) -> None:
        self.out = out
        self.board = chess.Board()
        self.searcher = Searcher()
        self.multipv = 1
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def send(self, line: str) -> None:
        with self._lock:
            self.out.write(line + "\n")
            self.out.flush()

    # --- commandes ---------------------------------------------------------------

    def handle(self, line: str) -> bool:
        """Traite une ligne reçue ; renvoie ``False`` sur « quit »."""
        tokens = line.split()
        if not tokens:
            return True
        command, args = tokens[0], tokens[1:]
        if command == "uci":
            self.send(f"id name {NAME} {VERSION}")
            self.send("id author Chessio")
            self.send(f"option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 4096")
            self.send("option name MultiPV type spin default 1 min 1 max 10")
            self.send("uciok")
        elif command == "isready":
            self.send("readyok")
        elif command == "ucinewgame":
            self._stop_search()
            self.searcher.clear()
            self.board = chess.Board()
        elif command == "setoption":
            self._set_option(args)
        elif command == "position":
            self._stop_search()
            self._set_position(args)
        elif command == "go":
            self._stop_search()
            if args and args[0] == "perft":
                self._perft(int(args[1]) if len(args) > 1 else 1)
            else:
                self._go(parse_limits(args))
        elif command == "stop":
            self._stop_search()
        elif command == "quit":
            self._stop_search()
            return False
        elif command == "d":
            self.send(str(self.board))
            self.send(f"Fen: {self.board.fen()}")
        return True

    def _set_option(self, args: list[str]) -> None:
        text = " ".join(args)
        if not text.startswith("name ") or " value " not in text:
            return
        name, value = text[len("name ") :].split(" value ", 1)
        name = name.strip().lower()
        try:
            number = int(value.strip())
        except ValueError:
            return
        if name == "hash":
            self.searcher.tt_entries = max(1, number) * ENTRIES_PER_MB
            self.searcher.tt.clear()
        elif name == "multipv":
            self.multipv = max(1, min(number, 10))

    def _set_position(self, args: list[str]) -> None:
        if not args:
            return
        moves: list[str] = []
        if "moves" in args:
            index = args.index("moves")
            args, moves = args[:index], args[index + 1 :]
        if args[0] == "startpos":
            board = chess.Board()
        elif args[0] == "fen":
            board = chess.Board(" ".join(args[1:]))
        else:
            return
        for move in moves:
            board.push_uci(move)
        self.board = board

    def _perft(self, depth: int) -> None:
        start = time.perf_counter()
        counts = divide(self.board.copy(), depth)
        for move, count in counts.items():
            self.send(f"{move}: {count}")
        elapsed = time.perf_counter() - start
        self.send("")
        self.send(f"Nodes searched: {sum(counts.values())}")
        self.send(f"Time: {elapsed * 1000:.0f} ms")

    # --- recherche ----------------------------------------------------------------

    def _go(self, limits: Limits) -> None:
        board = self.board.copy()
        multipv = self.multipv

        def run() -> None:
            info = self.searcher.search(board, limits, multipv=multipv, on_iteration=self._report)
            if limits.infinite:
                # UCI : en analyse infinie, « bestmove » n'est envoyé qu'après « stop ».
                while not self.searcher.stop_requested:
                    time.sleep(0.005)
            best = info.best
            ponder = info.lines[0].moves[1] if info.lines and len(info.lines[0].moves) > 1 else None
            self.send(
                f"bestmove {best.uci() if best else '0000'}"
                + (f" ponder {ponder.uci()}" if ponder else "")
            )

        self._thread = threading.Thread(target=run, name="recherche", daemon=True)
        self._thread.start()

    def _report(self, info: SearchInfo) -> None:
        nps = int(info.nodes / info.elapsed) if info.elapsed > 0 else 0
        for index, line in enumerate(info.lines, start=1):
            pv = " ".join(move.uci() for move in line.moves)
            self.send(
                f"info depth {info.depth} multipv {index} score {uci_score(line.score)} "
                f"nodes {info.nodes} nps {nps} time {int(info.elapsed * 1000)} pv {pv}"
            )

    def _stop_search(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self.searcher.stop()
            self._thread.join()
        self._thread = None


def main() -> int:
    engine = UciEngine()
    while True:
        line = sys.stdin.readline()
        if not line or not engine.handle(line.strip()):
            break
    return 0
