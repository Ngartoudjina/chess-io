"""Recette du lot L1 : reconnaissance de vraies captures Chess.com (20 positions).

Chaque position est chargée dans l'échiquier d'analyse de Chess.com à partir
d'une FEN connue, puis capturée : l'étiquette de chaque case est donc connue
d'avance (cahier des charges, section 8).

    python scripts/evaluer_captures.py --instructions img/A_CAPTURER.md
    python scripts/evaluer_captures.py --calibration img/00.png --dossier img
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import chess

from chessio.recognize import Orientation, recognize
from chessio.vision.board import crop_board, load_image, locate_board, split_cells
from chessio.vision.templates import TemplateSet, calibrate


@dataclass(frozen=True)
class Sample:
    number: int
    name: str
    start_fen: str  # FEN à charger dans Chess.com
    move: str | None  # coup à jouer ensuite (SAN), pour obtenir le surlignage
    black_bottom: bool  # retourner l'échiquier (Noirs en bas)

    @property
    def filename(self) -> str:
        return f"{self.number:02d}.png"

    @property
    def expected(self) -> chess.Board:
        board = chess.Board(self.start_fen)
        if self.move:
            board.push_san(self.move)
        return board


def _from_moves(number, name, san: str, *, highlight=False, black_bottom=False) -> Sample:
    board = chess.Board()
    moves = san.split()
    for move in moves[:-1] if highlight else moves:
        board.push_san(move)
    return Sample(number, name, board.fen(), moves[-1] if highlight else None, black_bottom)


SAMPLES = (
    _from_moves(
        1,
        "Espagnole, défense berlinoise",
        "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8",
        highlight=True,
    ),
    _from_moves(
        2,
        "Sicilienne Najdorf",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3",
        black_bottom=True,
    ),
    _from_moves(
        3,
        "Gambit dame refusé",
        "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6",
        highlight=True,
    ),
    _from_moves(
        4,
        "Est-indienne",
        "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7",
    ),
    _from_moves(
        5,
        "Caro-Kann",
        "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3",
        highlight=True,
        black_bottom=True,
    ),
    _from_moves(6, "Française Winawer", "e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4"),
    _from_moves(7, "Italienne", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O a4 a5"),
    _from_moves(
        8,
        "Système de Londres",
        "d4 d5 Bf4 Nf6 e3 e6 Nf3 c5 c3 Nc6 Nbd2 Bd6 Bg3 O-O Bd3 b6 Ne5",
        highlight=True,
        black_bottom=True,
    ),
    _from_moves(9, "Gambit Evans", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O"),
    _from_moves(
        10,
        "Sicilienne Dragon",
        "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4 Bd7 O-O-O Rc8 Bb3 Ne5",
        highlight=True,
    ),
    Sample(11, "Finale de Lucena", "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1", None, False),
    Sample(12, "Roi et pion contre roi", "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", None, True),
    Sample(13, "Dame contre tour", "8/8/8/3k4/8/8/2r5/K3Q3 w - - 0 1", None, False),
    Sample(14, "Deux dames", "6k1/5ppp/8/8/8/8/1Q3PPP/1Q4K1 b - - 0 1", None, True),
    Sample(
        15,
        "Milieu de partie, fous",
        "r4rk1/pp3ppp/2n1b3/2b1p3/4P3/2N1BN2/PPP2PPP/R4RK1 w - - 0 1",
        "Bxc5",
        False,
    ),
    Sample(
        16,
        "Italienne symétrique",
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
        None,
        True,
    ),
    Sample(17, "Fou et cavalier contre roi", "8/8/4k3/8/8/2B2N2/8/4K3 w - - 0 1", None, False),
    Sample(
        18,
        "Milieu de partie chargé",
        "r2q1rk1/1b2bppp/p1n1pn2/1pp5/3P4/1BN1PN2/PP3PPP/R1BQR1K1 b - - 0 1",
        None,
        True,
    ),
    Sample(19, "Course de pions", "8/5k2/8/1p6/8/8/6PK/8 b - - 0 1", None, False),
    Sample(20, "Mat à l'étouffée", "6rk/5Npp/8/8/8/8/6PP/6K1 b - - 0 1", None, False),
)


def write_instructions(path: Path) -> None:
    lines = [
        "# Captures à réaliser pour la recette du lot L1",
        "",
        "Pour chaque ligne : sur chess.com/analysis, chargez la FEN (champ FEN/PGN),",
        "jouez le coup indiqué s'il y en a un (il sera surligné), retournez l'échiquier",
        "si demandé (Noirs en bas), puis capturez l'écran avec l'outil d'Ubuntu",
        "(Impr. écran) — plein écran ou zone, peu importe, mais l'échiquier entier.",
        "Enregistrez l'image dans ce dossier sous le nom indiqué.",
        "",
        "Gardez le même thème que `00.png` (position initiale, Blancs en bas).",
        "",
        "| Fichier | Position | FEN à charger | Coup à jouer | Noirs en bas |",
        "|---|---|---|---|---|",
    ]
    for s in SAMPLES:
        lines.append(
            f"| `{s.filename}` | {s.name} | `{s.start_fen}` | {s.move or '—'} | "
            f"{'**oui**' if s.black_bottom else 'non'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Instructions écrites : {path}")


def evaluate(folder: Path, templates: TemplateSet) -> int:
    exact = squares_ok = squares_total = missing = oriented_wrong = 0
    timings = []
    sources: dict[str, int] = {}
    for s in SAMPLES:
        image_path = folder / s.filename
        if not image_path.exists():
            missing += 1
            continue
        expected = s.expected
        try:
            # Comme en usage réel : orientation automatique, seule la couleur au trait est donnée.
            result = recognize(
                load_image(image_path), expected.turn, templates, orientation=Orientation.AUTO
            )
        except ValueError as exc:
            print(f"{s.filename}  ÉCHEC  {exc}")
            squares_total += 64
            continue
        timings.append(result.elapsed)
        if result.white_bottom == s.black_bottom:
            oriented_wrong += 1
            print(
                f"{s.filename}  KO   orientation fausse "
                f"({'Blancs' if result.white_bottom else 'Noirs'} en bas, "
                f"d'après : {result.orientation_source})  {s.name}"
            )
            squares_total += 64
            continue
        sources[result.orientation_source] = sources.get(result.orientation_source, 0) + 1
        note = ""
        before = chess.Board(s.start_fen)
        if s.move and result.board.board_fen() == before.board_fen():
            # La capture montre la position avant le coup demandé : coup non joué
            # sur Chess.com. On juge la lecture sur ce que montre l'image.
            expected, note = before, "  (coup non joué : position avant le coup)"
        wrong = [sq for sq in chess.SQUARES if result.board.piece_at(sq) != expected.piece_at(sq)]
        squares_ok += 64 - len(wrong)
        squares_total += 64
        exact += not wrong
        status = ("OK " if not wrong else "KO ") + note
        details = ", ".join(
            f"{chess.square_name(sq)}: lu {result.board.piece_at(sq) or 'vide'} "
            f"au lieu de {expected.piece_at(sq) or 'vide'}"
            for sq in wrong
        )
        doubtful = ", ".join(chess.square_name(sq) for sq in result.doubtful)
        print(
            f"{s.filename}  {status} {result.elapsed * 1000:4.0f} ms  {s.name}"
            + (f"\n      erreurs : {details}" if details else "")
            + (f"\n      douteuses : {doubtful}" if doubtful else "")
        )
    done = len(SAMPLES) - missing
    if not done:
        print(f"Aucune capture trouvée dans {folder} (fichiers 01.png à 20.png).")
        return 1
    print(
        f"\nPositions exactes : {exact}/{done}   "
        f"cases exactes : {squares_ok}/{squares_total} "
        f"({100 * squares_ok / squares_total:.2f} %)   "
        f"temps max : {max(timings, default=0) * 1000:.0f} ms"
        + (f"   ({missing} captures manquantes)" if missing else "")
    )
    print(
        f"Orientation : {done - oriented_wrong}/{done} justes ("
        + ", ".join(f"{n} par {source}" for source, n in sorted(sources.items()))
        + ")"
    )
    return 0 if exact == done else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--instructions", type=Path, help="écrire la liste des captures à faire")
    parser.add_argument("--calibration", type=Path, help="capture de la position initiale")
    parser.add_argument("--dossier", type=Path, default=Path("img"), help="dossier des captures")
    args = parser.parse_args()

    if args.instructions:
        write_instructions(args.instructions)
        return 0
    if args.calibration is None:
        parser.error("--calibration est requis pour évaluer")
    image = load_image(args.calibration)
    templates = calibrate(split_cells(crop_board(image, locate_board(image))))
    return evaluate(args.dossier, templates)


if __name__ == "__main__":
    sys.exit(main())
