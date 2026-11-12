"""Interface en ligne de commande du lot L0 : FEN → meilleur coup.

Exemples ::

    chessio analyse "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    chessio analyse rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR --trait b --multipv 3
"""

from __future__ import annotations

import argparse
import sys

import chess

from chessio import __version__
from chessio.engine import DEFAULT_TIME, Engine, EngineNotFoundError
from chessio.position import InvalidPositionError, build_board, parse_turn, validate

WARNING = (
    "Avertissement : à utiliser pour l'analyse de parties terminées, les parties contre "
    "les bots, les problèmes et l'étude — jamais pendant une partie contre un humain."
)


def _board_from_args(fen: str, turn: str | None) -> chess.Board:
    fields = fen.split()
    if len(fields) == 1:
        if turn is None:
            raise ValueError("Placement seul : indiquez la couleur au trait avec --trait w|b")
        return build_board(fields[0], turn)
    board = chess.Board(fen)
    if turn is not None:
        board.turn = parse_turn(turn)
    return board


def _cmd_analyse(args: argparse.Namespace) -> int:
    board = _board_from_args(args.fen, args.trait)
    issues = validate(board)
    if issues:
        print("Position invalide, aucun coup proposé :", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 2

    trait = "Blancs" if board.turn == chess.WHITE else "Noirs"
    print(f"FEN   : {board.fen()}")
    print(f"Trait : {trait}")
    with Engine(args.moteur) as engine:
        analysis = engine.analyse(board, time_limit=args.temps, multipv=args.multipv)
        print(f"Moteur: {engine.name} ({analysis.elapsed:.2f} s)")

    if analysis.outcome:
        print(f"Partie terminée : {analysis.outcome}.")
        return 0
    for rank, line in enumerate(analysis.lines, start=1):
        print(
            f"{rank}. {line.san:<7} ({line.uci})  éval {line.evaluation:>6}  "
            f"prof. {line.depth:<3} {line.pv_san}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessio", description=WARNING)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyse = sub.add_parser("analyse", help="meilleur coup pour une FEN")
    analyse.add_argument("fen", help="FEN complète, ou placement seul avec --trait")
    analyse.add_argument("--trait", choices=["w", "b"], help="couleur au trait")
    analyse.add_argument(
        "--temps", type=float, default=DEFAULT_TIME, help="secondes de réflexion (0,1 à 10)"
    )
    analyse.add_argument("--multipv", type=int, default=1, help="nombre de coups proposés")
    analyse.add_argument("--moteur", help="chemin d'un moteur UCI (Stockfish par défaut)")
    analyse.set_defaults(func=_cmd_analyse)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, InvalidPositionError, EngineNotFoundError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
