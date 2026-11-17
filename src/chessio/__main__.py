"""Interface en ligne de commande : FEN ou capture → meilleur coup.

Exemples ::

    chessio analyse "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    chessio analyse rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR --trait b --multipv 3
    chessio calibrer position_initiale.png
    chessio capture partie.png --trait w
    chessio gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _print_issues(issues) -> int:
    print("Position invalide, aucun coup proposé :", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    return 2


def _cmd_analyse(args: argparse.Namespace) -> int:
    board = _board_from_args(args.fen, args.trait)
    issues = validate(board)
    if issues:
        return _print_issues(issues)
    return _analyse_board(board, args)


def _analyse_board(board: chess.Board, args: argparse.Namespace) -> int:
    trait = "Blancs" if board.turn == chess.WHITE else "Noirs"
    print(f"FEN   : {board.fen()}")
    print(f"Trait : {trait}")
    with Engine.open(args.moteur) as engine:
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


def _cmd_calibrer(args: argparse.Namespace) -> int:
    from chessio.paths import templates_path
    from chessio.vision.board import crop_board, load_image, locate_board, split_cells
    from chessio.vision.templates import calibrate

    image = load_image(args.image)
    region = locate_board(image)
    templates = calibrate(split_cells(crop_board(image, region)))
    path = args.sortie or templates_path()
    templates.save(path)
    print(f"Plateau trouvé : x={region.x} y={region.y} côté={region.size} px")
    print(f"Calibration enregistrée : {path}")
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    from chessio.paths import templates_path
    from chessio.recognize import Orientation, recognize
    from chessio.vision.board import load_image
    from chessio.vision.templates import TemplateSet

    path = args.gabarits or templates_path()
    templates = TemplateSet.load(path) if path.exists() else None  # calibration facultative
    result = recognize(
        load_image(args.image),
        args.trait,
        templates,
        orientation=Orientation(args.orientation),
    )
    print(f"Reconnaissance : {result.elapsed * 1000:.0f} ms")
    bottom = "Blancs" if result.white_bottom else "Noirs"
    print(f"Orientation : {bottom} en bas (d'après : {result.orientation_source})")
    if args.trait is None and not result.turn_detected:
        print(
            "Trait non détecté (aucun dernier coup surligné ni roi en échec) : "
            "précisez-le avec --trait w|b",
            file=sys.stderr,
        )
        return 2
    if result.turn_detected:
        mover = "Blancs" if result.board.turn == chess.WHITE else "Noirs"
        print(f"Trait détecté : {mover} (d'après : {result.turn_source})")
    for note in result.notes:
        print(f"Remarque : {note}", file=sys.stderr)
    if not result.ok:
        print(f"FEN lue : {result.board.fen()}", file=sys.stderr)
        return _print_issues(result.issues)
    if args.sans_moteur:
        print(f"FEN   : {result.board.fen()}")
        return 0
    return _analyse_board(result.board, args)


def _cmd_match(args: argparse.Namespace) -> int:
    from chessio.match import Match, MatchSettings

    settings = MatchSettings(
        tested=args.moteur,
        reference=args.adversaire,
        elo=args.elo,
        games=args.parties,
        seconds=args.temps,
    )
    if not args.sans_fenetre:
        from chessio.ui.app import create_app
        from chessio.ui.match import MatchWindow

        app = create_app([])
        window = MatchWindow(settings, delay=args.delai, pgn=args.pgn)
        window.resize(1320, 880)
        window.show()
        return app.exec()

    def on_game(result) -> None:
        side = "Blancs" if result.tested_white else "Noirs"
        print(
            f"partie {result.number:2d} : {result.result:7s} ({side}, {result.termination}, "
            f"{result.plies} demi-coups)",
            flush=True,
        )
        if args.pgn:
            with args.pgn.open("a", encoding="utf-8") as fh:
                print(result.game, file=fh, end="\n\n")

    summary = Match(settings).run(on_game=on_game)
    print(f"\n{summary.tested} contre {summary.reference}")
    print(summary.text())
    return 0


def _cmd_jouer(args: argparse.Namespace) -> int:
    from chessio.ui.app import create_app
    from chessio.ui.play import PlayWindow

    app = create_app([])
    window = PlayWindow()
    window.resize(1280, 840)
    window.show()
    return app.exec()


def _cmd_installer(args: argparse.Namespace) -> int:
    from chessio.desktop import install

    for path in install():
        print(f"Installé : {path}")
    print("Chessio est dans le menu des applications (Jeux), avec son icône.")
    return 0


def _cmd_desinstaller(args: argparse.Namespace) -> int:
    from chessio.desktop import uninstall

    removed = uninstall()
    print("\n".join(f"Retiré : {p}" for p in removed) or "Rien à retirer.")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from chessio.ui.app import main as gui_main

    return gui_main()


def _add_engine_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temps", type=float, default=DEFAULT_TIME, help="secondes de réflexion (0,1 à 10)"
    )
    parser.add_argument("--multipv", type=int, default=1, help="nombre de coups proposés")
    parser.add_argument(
        "--moteur",
        help="« maison » pour le moteur maison, ou chemin d'un moteur UCI (Stockfish par défaut)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chessio", description=WARNING)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")  # sans commande : l'interface

    analyse = sub.add_parser("analyse", help="meilleur coup pour une FEN")
    analyse.add_argument("fen", help="FEN complète, ou placement seul avec --trait")
    analyse.add_argument("--trait", choices=["w", "b"], help="couleur au trait")
    _add_engine_options(analyse)
    analyse.set_defaults(func=_cmd_analyse)

    calibrer = sub.add_parser("calibrer", help="apprendre le thème sur la position initiale")
    calibrer.add_argument("image", help="capture de la position initiale (PNG ou JPEG)")
    calibrer.add_argument("--sortie", type=Path, help="fichier de gabarits à écrire")
    calibrer.set_defaults(func=_cmd_calibrer)

    capture = sub.add_parser("capture", help="meilleur coup à partir d'une capture")
    capture.add_argument("image", help="capture d'écran (PNG ou JPEG)")
    capture.add_argument(
        "--trait",
        choices=["w", "b"],
        help="couleur au trait (par défaut : déduite du dernier coup surligné)",
    )
    capture.add_argument(
        "--orientation",
        choices=["auto", "w", "b"],
        default="auto",
        help="camp affiché en bas (auto : d'après les coordonnées)",
    )
    capture.add_argument("--gabarits", type=Path, help="fichier de calibration")
    capture.add_argument("--sans-moteur", action="store_true", help="afficher la FEN seulement")
    _add_engine_options(capture)
    capture.set_defaults(func=_cmd_capture)

    match = sub.add_parser("match", help="match entre deux moteurs, en direct à l'écran")
    match.add_argument("--moteur", default="maison", help="moteur testé (« maison » par défaut)")
    match.add_argument(
        "--adversaire", default="stockfish", help="adversaire (« stockfish » ou chemin UCI)"
    )
    match.add_argument(
        "--elo", type=int, help="Elo imposé à l'adversaire (Stockfish : 1320 à 3190)"
    )
    match.add_argument("--parties", type=int, default=20, help="nombre de parties")
    match.add_argument("--temps", type=float, default=0.3, help="secondes de réflexion par coup")
    match.add_argument("--delai", type=float, default=0.5, help="pause d'affichage entre les coups")
    match.add_argument("--pgn", type=Path, help="fichier où ajouter les parties jouées")
    match.add_argument("--sans-fenetre", action="store_true", help="résultats en texte seulement")
    match.set_defaults(func=_cmd_match)

    jouer = sub.add_parser("jouer", help="jouer contre l'IA, avec indices")
    jouer.set_defaults(func=_cmd_jouer)

    installer = sub.add_parser("installer", help="ajouter Chessio au menu des applications")
    installer.set_defaults(func=_cmd_installer)
    desinstaller = sub.add_parser("desinstaller", help="retirer Chessio du menu")
    desinstaller.set_defaults(func=_cmd_desinstaller)

    gui = sub.add_parser("gui", help="interface graphique")
    gui.set_defaults(func=_cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        return _cmd_gui(args)
    try:
        return args.func(args)
    except (ValueError, InvalidPositionError, EngineNotFoundError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
